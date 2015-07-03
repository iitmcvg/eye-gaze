all:
	oic
oic:
	
	g++ -lX11 ./dlib/all/source.cpp oic.cpp faceDetection.cpp kalmanFilters.cpp util.cpp pupilDetection.cpp viewUtils.cpp -o oic -ljpeg -mavx `pkg-config --cflags --libs opencv`

