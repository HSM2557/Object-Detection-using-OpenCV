TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += /usr/local/include

include(opencv31.pri)
include(RaspberryPiCamera.pri)

SOURCES += main.cpp
