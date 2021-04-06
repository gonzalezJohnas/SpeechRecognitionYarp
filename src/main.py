import librosa
import speech_recognition as sr

import sys
import os
import time
import yarp
import numpy as np
import soundfile as sf

from utils import frame_generator, read_wave


def info(msg):
    print("[INFO] {}".format(msg))

LANGUAGE_CODE = ["it-IT, en-EN, es-ES, fr-FR"]

class SpeechToTextModule(yarp.RFModule):
    """
    Description:
        Class to perform speech recognition via Google-Cloud API

    Args:
        input_port  : Audio from remoteInterface

    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # Voice activity parameters
        self.threshold_voice = None
        self.voice_detected = False

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)
        self.module_name = None
        self.saving_path = None

        # Define vars to receive an sound
        self.audio_in_port = yarp.BufferedPortSound()
        self.audio_power_port = yarp.Port()

        self.sample_rate = 48000
        self.sound = yarp.Sound()
        self.audio = []
        self.record = True
        self.np_audio = None

        # Define a port to send a speech
        self.speech_output = yarp.Port()
        self.actions_output = yarp.Port()
        self.actions_list = {}

        # Speech-recognition parameters
        self.speech_recognizer = None
        self.language = "it-IT"

    def configure(self, rf):

        # Module parameters
        self.module_name = rf.check("name",
                                    yarp.Value("SpeechToText"),
                                    "module name (string)").asString()

        self.saving_path = rf.check("path",
                                    yarp.Value("/tmp"),
                                    "saving path name (string)").asString()
        self.language  = rf.check("lang",
                                    yarp.Value("it-IT"),
                                    "Language for the Google speech API (string)").asString()


        # Opening Ports
        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Audio
        self.audio_in_port.open('/' + self.module_name + '/speech:i')
        self.audio_power_port.open('/' + self.module_name + '/power:i')

        # Speech
        self.speech_output.open('/' + self.module_name + '/speech:o')
        self.actions_output.open('/' + self.module_name + '/actions:o')


        # Speech recognition parameters
        self.threshold_voice = rf.check("voice_threshold",
                                    yarp.Value("4"),
                                    "Energy threshold use by the VAD (int)").asDouble()

        self.speech_recognizer = sr.Recognizer()

        # Actions list
        keywords = rf.find("KEYWORDS").asList()
        actions = rf.find("ACTIONS").asList()
        self.process_actions(keywords, actions)

        info("Initialization complete")

        return True

    def process_actions(self, list_keywords, list_actions):

        for i in range(0, list_actions.size()):
            k = list_keywords.get(i).asString()
            a = list_actions.get(i).asString()
            self.actions_list[k] = a


    def interruptModule(self):
        info("stopping the module")
        self.audio_in_port.interrupt()
        self.handle_port.interrupt()
        self.speech_output.interrupt()
        self.actions_output.interrupt()

        return True

    def close(self):
        self.audio_in_port.close()
        self.handle_port.close()
        self.speech_output.close()
        self.actions_output.close()

        return True

    def respond(self, command, reply):
        ok = False

        # Is the command recognized
        rec = False

        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False

        elif command.get(0).asString() == "help":
            reply.addVocab(yarp.encode("many"))
            reply.addString("Speech recognition module commands are")

            reply.addString("start : Start the recording")
            reply.addString("stop : Stop the recording")
            reply.addString("set lang : Change the language for the Google Speech API")
            reply.addString("set thr : Set the threshold for the VAD")

        elif command.get(0).asString() == "start":
            if self.audio_in_port.getInputCount():
                self.audio = []
                self.record = True
                info("starting recording!")

                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "stop":
            info("stopping recording!")
            self.record = False

            reply.addString("ok")

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                self.threshold_voice = command.get(2).asDouble()
                reply.addString("ok")
            if command.get(1).asString() == "lang":
                lang_to_set = command.get(2).asString()
                if lang_to_set in LANGUAGE_CODE:
                    self.language = lang_to_set
                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                reply.addDouble(self.threshold_voice)

            else:
                reply.addString("nack")
        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.05

    def record_audio(self):
        self.sound = self.audio_in_port.read(False)
        if self.sound:

            chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)

            for c in range(self.sound.getChannels()):
                for i in range(self.sound.getSamples()):
                    chunk[c][i] = self.sound.get(i, c) / 32768.0

            self.audio.append(chunk)


    def get_power(self):
        max_power = 0.0
        if self.audio_power_port.getInputCount():

            power_matrix = yarp.Matrix()
            self.audio_power_port.read(power_matrix)
            power_values = [power_matrix[0, 1], power_matrix[0, 0]]
            max_power = np.max(power_values)
            info("Max power is {}".format(max_power))

        return max_power

    def updateModule(self):

        if self.record:

            self.record_audio()
            audio_power = self.get_power()

            if audio_power > self.threshold_voice:
                if not self.voice_detected:
                    self.audio = self.audio[-10:]
                    self.voice_detected = True
                info("Voice detected")

            elif audio_power < (self.threshold_voice/2) and self.voice_detected:

                info(" Stop voice")
                np_audio = np.concatenate(self.audio, axis=1)
                np_audio = librosa.util.normalize(np_audio, axis=1)
                np_audio = np.squeeze(np_audio)
                signal = np.transpose(np_audio, (1, 0))
                sf.write(self.saving_path + '/speech.wav', signal, self.sample_rate)

                recognized_text = self.speech_to_text()
                if len(recognized_text):
                    self.write_text(recognized_text)
                    self.recognize_actions(recognized_text)

                self.voice_detected = False
                self.audio = []

        return True

    def speech_to_text(self):
        with sr.AudioFile(self.saving_path + "/speech.wav") as source:
            audio = self.speech_recognizer.record(source)  # read the entire audio file

        transcript = ""
        try:
            transcript = self.speech_recognizer.recognize_google(audio, language=self.language, )
        except sr.UnknownValueError:
            info("Google Speech Recognition could not understand audio")

        info("Google Speech Recognition thinks you said " + transcript)

        return transcript

    def recognize_actions(self, input_speech):
        for keyword, action_name in self.actions_list.items():
            if keyword in input_speech.lower():
                return self.send_actions(action_name)

    def write_text(self, text):
        if self.speech_output.getOutputCount():
            action_bottle = yarp.Bottle()
            action_bottle.clear()

            action_bottle.addString(text)
            self.speech_output.write(action_bottle)


    def send_actions(self, actions_name):
        if self.actions_output.getOutputCount():
            action_bottle = yarp.Bottle()
            action_bottle.clear()

            action_bottle.addString('exe')
            action_bottle.addString(actions_name)

            self.actions_output.write(action_bottle)

            return True
        return False



if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        info("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    speechToTextModule = SpeechToTextModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('speechToText')
    rf.setDefaultConfigFile('speechToText.ini')

    if rf.configure(sys.argv):
        speechToTextModule.runModule(rf)
    sys.exit()
