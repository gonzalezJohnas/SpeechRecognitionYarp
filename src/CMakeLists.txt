cmake_minimum_required(VERSION 3.6)
set(KEYWORD SpeechRecognition)
set(NAME_VENV ${KEYWORD}_venv)
project($KEYWORD)

set(PROGRAM_PERMISSIONS_DEFAULT
        OWNER_WRITE OWNER_READ OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

# Find Python and Virtualenv.
find_program(VIRTUALENV virtualenv)

if(NOT VIRTUALENV)
    message(FATAL_ERROR "Could not find `virtualenv` in PATH")
endif()

set(VIRTUALENV ${VIRTUALENV})

# Find iCubCONTRIB packages
FIND_PACKAGE(ICUBcontrib REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
include(ICUBcontribHelpers)
include(ICUBcontribOptions)

icubcontrib_set_default_prefix()
set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")


# Generate the virtualenv and ensure it's up to date.
add_custom_command(
        OUTPUT ${NAME_VENV}
        COMMAND ${VIRTUALENV} -p /usr/bin/python3 ${NAME_VENV}
)
add_custom_command(
        OUTPUT requirements
        DEPENDS ${NAME_VENV}
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt  ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
        COMMAND ./${NAME_VENV}/bin/pip install -r requirements.txt
)

# Generate the binary using pyinstaller and put the binary in the bin folder
add_custom_command(
        OUTPUT binary
        DEPENDS ${NAME_VENV} requirements
        COMMAND ./${NAME_VENV}/bin/pyinstaller  --onefile ${CMAKE_CURRENT_SOURCE_DIR}/main.py --distpath bin -n SpeechRecognition

)


# Custom targets for CMake
# Create a virtual-env and install all requirements
add_custom_target(build-python  ALL
        DEPENDS binary
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
        )



install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/${KEYWORD} PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT} DESTINATION bin)