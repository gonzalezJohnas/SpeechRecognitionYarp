from numpy.core.fromnumeric import std
import librosa
import speech_recognition as sr

import sys
import os
import time
import yarp
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wavfile

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
        self.threshold_voice_low = None
        self.calib = False
        self.stop_voice_counter = 0

        self.voice_detected = False

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)
        self.module_name = None
        self.saving_path = None

        # Define vars to receive an sound
        self.audio_in_port = yarp.BufferedPortSound()
        self.audio_power_port = yarp.BufferedPortBottle()

        self.sample_rate = 48000
        self.sound = yarp.Sound()
        self.audio = []
        self.record = None
        self.np_audio = None

        # Define a port to send a speech
        self.speech_output = yarp.Port()
        self.actions_output = yarp.Port()
        self.emotions_output = yarp.Port()

        self.actions_list = {}

        # Speech-recognition parameters
        self.speech_recognizer = None
        self.language = ""

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

        self.audio_in_port.setStrict(True)

        # Speech
        self.speech_output.open('/' + self.module_name + '/speech:o')

        # Actions and emotions
        self.actions_output.open('/' + self.module_name + '/actions:o')
        self.emotions_output.open('/' + self.module_name + '/emotions:o')


        # Speech recognition parameters
        self.threshold_voice = rf.check("voice_threshold",
                                    yarp.Value("4"),
                                    "Energy threshold use by the VAD (int)").asDouble()
        self.threshold_voice_low = self.threshold_voice / 2

        self.language = rf.check("lang",
                                    yarp.Value("it-IT"),
                                    "Energy threshold use by the VAD (int)").asString()

        self.record = rf.check("auto_start",
                                    yarp.Value(True),
                                    "Start the module automatically (bool)").asBool()


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
                self.audio_in_port.resume()

                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "stop":
            info("stopping recording!")
            self.record = False
            self.audio_in_port.interrupt()
            reply.addString("ok")

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                self.threshold_voice = command.get(2).asDouble()
                self.threshold_voice_low = self.threshold_voice / 2

                reply.addString("ok")
            elif command.get(1).asString() == "lang":
                self.language = command.get(2).asString()
                reply.addString("ok")
            if command.get(1).asString() == "lang":
                lang_to_set = command.get(2).asString()
                if lang_to_set in LANGUAGE_CODE:
                    self.language = lang_to_set
                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "calib":
            self.calib = True         
            reply.addString("ok")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                reply.addDouble(self.threshold_voice)
            elif command.get(1).asString() == "lang":
                reply.addString(self.language)
            else:
                reply.addString("nack")
        
        else:
            reply.addString("nack")
        
        
        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.01

    def record_audio(self, blocking=False):
        while(self.audio_in_port.getPendingReads()):
            self.sound = self.audio_in_port.read(blocking)
            if self.sound:

                chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)

                for c in range(self.sound.getChannels()):
                    for i in range(self.sound.getSamples()):
                        chunk[c][i] = self.sound.get(i, c) / 32768.0

                self.audio.append(chunk)


    def calibrate_power(self):

        power_values = np.zeros(50, dtype=np.float32)
        val_read = 0

        while(val_read < 50):

            power = self.get_power()
            if power > 0:

                power_values[val_read] = power

                val_read += 1
        
        mean_power = power_values.mean()
        std_power = power_values.std()
        
        self.threshold_voice = mean_power + 10 *std_power
        self.threshold_voice_low = mean_power 

        info("Mean stds: {} {}".format(mean_power, std_power))
        info("New threshold values: {} {}".format(self.threshold_voice_low, self.threshold_voice))


    def get_power(self):
        max_power = -1
        if self.audio_power_port.getInputCount():
            power_matrix = self.audio_power_port.read(False)
            if power_matrix:

                power_matrix = power_matrix.get(2).asList()
                power_values = [power_matrix.get(0).asDouble(), power_matrix.get(1).asDouble()]
                max_power = np.max(power_values)
                if max_power < self.threshold_voice_low and self.voice_detected:
                    self.stop_voice_counter += 1

        return max_power

    def updateModule(self):
       
        if self.record:
            if self.calib:
                self.calibrate_power()
                self.calib = False


            self.record_audio()

      
            audio_power = self.get_power()
            info("Max power is {}".format(audio_power))

            if audio_power > self.threshold_voice and not self.voice_detected:
                self.audio = self.audio[-4:]
                self.voice_detected = True
                info("Voice detected")

            elif self.stop_voice_counter > 10 and self.voice_detected:
                self.send_emotions('hap')
                self.stop_voice_counter = 0
                self.record_audio()


                info(" Stop voice")
                np_audio = np.concatenate(self.audio, axis=1)
                np_audio = librosa.util.normalize(np_audio, axis=1)
                np_audio = np.squeeze(np_audio)
                signal = np.transpose(np_audio, (1, 0))
                #sf.write(self.saving_path + '/speech.wav', signal, self.sample_rate)
                signal = (np.iinfo(np.int32).max * (signal/np.abs(signal).max())).astype(np.int32)

                wavfile.write(self.saving_path + '/speech.wav', self.sample_rate, signal)
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
            self.send_emotions('cun')

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

    def send_emotions(self, emotion):
        if self.emotions_output.getOutputCount():
            emo_bottle = yarp.Bottle()
            emo_bottle.clear()

            emo_bottle.addString('set')
            emo_bottle.addString('all')
            emo_bottle.addString(emotion)


            self.emotions_output.write(emo_bottle)

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
