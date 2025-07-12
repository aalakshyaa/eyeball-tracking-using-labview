#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
using namespace std;
using namespace cv;

class My_face_eye_detection {  
public:	
	// classifier object for face
	CascadeClassifier face;
	// allocate for the classification results
	vector<Rect> detected_face;
	// allocate for results of rectangles of detected face(s) to pass
	vector<int> x1_face,y1_face,x2_face,y2_face;	
	
	// classifier object for eyes
	CascadeClassifier eyes;
	// allocate for the classification results
	vector<Rect> detected_eyes;
	// allocate for results of rectangles of detected eyes to pass
	vector<int> x1_eyes,y1_eyes,x2_eyes,y2_eyes;
public:
	// setup detection
	int detection_setup () {
		face.load("C:\\Program Files (x86)\\OpenCV 2.4.6\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml");
		eyes.load("C:\\Program Files (x86)\\OpenCV 2.4.6\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");

		if(!face.load("C:\\Program Files (x86)\\OpenCV 2.4.6\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml") || 
		   !eyes.load("C:\\Program Files (x86)\\OpenCV 2.4.6\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"))
			return -1;
		return 0;
	}

	// detect face and eyes
	int detection_start(Mat &image_in) {
		// check for image data
		if(!image_in.data) {
			return -1;
		}
		// detect face(s)
		face.detectMultiScale(image_in,detected_face,1.1,3,CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE,Size(30,30));
		if(detected_face.size() != 0) {
			// fill points vectors
			for (int i=0;i<detected_face.size();i++) {
				// fill vector of rectangle points for detected faces
				Rect r_face = detected_face[i];
				x1_face.push_back(r_face.tl().x);
				y1_face.push_back(r_face.tl().y);
				x2_face.push_back(r_face.br().x);
				y2_face.push_back(r_face.br().y);
				// limit ROI to detect eyes on the faces
				Mat faceROI = image_in(detected_face[i]);
				// detect eyes
				eyes.detectMultiScale(faceROI,detected_eyes,1.1,1,CV_HAAR_SCALE_IMAGE,Size(20,20));
				// for each face, detect eyes
				for (int j=0;j<detected_eyes.size();j++) {
					// fill vector of rectangle points for detected eyes
					Rect r_eyes = detected_eyes[j];
					x1_eyes.push_back(r_eyes.tl().x+r_face.tl().x);
					y1_eyes.push_back(r_eyes.tl().y+r_face.tl().y);
					x2_eyes.push_back(r_eyes.br().x+r_face.tl().x);
					y2_eyes.push_back(r_eyes.br().y+r_face.tl().y);
				}			
			}
			return 0;
		}

		else return -1;
	}

};

// extern C
extern "C" {	
	
	_declspec (dllexport) int detection_setup();
	_declspec (dllexport) int detection_start(uchar *imdata, int cols, int rows, int *result_size_face, int *result_size_eyes);
	_declspec (dllexport) int get_results(int *x1_face_out, int *y1_face_out, int *x2_face_out, int *y2_face_out, int *x1_eyes_out, int *y1_eyes_out, int *x2_eyes_out, int *y2_eyes_out);	
}

My_face_eye_detection faceEye_detector;


_declspec (dllexport) int detection_setup() 

{
	// load the classifier libraries
	int load_error = faceEye_detector.detection_setup();
	if(load_error != 0) return -1; 
	return 0;
}

_declspec (dllexport) int detection_start(uchar *imdata, 
										 int cols, 
										 int rows, 
										 int *result_size_face,
										 int *result_size_eyes)
{
	// create pointer to image data
	Mat image(rows,cols,CV_8U,&imdata[0]);
	// equalize histogram
	cv::equalizeHist(image,image);
	int detect_error = faceEye_detector.detection_start(image);
	if(detect_error != 0) {
		(*result_size_face) = 0;
		(*result_size_eyes) = 0;
		return -1;
	}
	// get result size
	(*result_size_face) = faceEye_detector.detected_face.size();
	(*result_size_eyes) = faceEye_detector.detected_eyes.size();	
	return 0;
}

_declspec (dllexport) int get_results(int *x1_face_out, 
								      int *y1_face_out,
									  int *x2_face_out, 
									  int *y2_face_out, 
									  int *x1_eyes_out, 
									  int *y1_eyes_out, 
									  int *x2_eyes_out, 
									  int *y2_eyes_out)
{	
	if(faceEye_detector.detected_face.size() != 0) {
		// copy rectangle points of detected faces
		memcpy(x1_face_out,&faceEye_detector.x1_face[0],faceEye_detector.detected_face.size()*sizeof(int));
		memcpy(y1_face_out,&faceEye_detector.y1_face[0],faceEye_detector.detected_face.size()*sizeof(int));
		memcpy(x2_face_out,&faceEye_detector.x2_face[0],faceEye_detector.detected_face.size()*sizeof(int));
		memcpy(y2_face_out,&faceEye_detector.y2_face[0],faceEye_detector.detected_face.size()*sizeof(int));

		// erase all vectors (first face and then eyes)
		faceEye_detector.x1_face.erase(faceEye_detector.x1_face.begin(),faceEye_detector.x1_face.end());
		faceEye_detector.y1_face.erase(faceEye_detector.y1_face.begin(),faceEye_detector.y1_face.end());
		faceEye_detector.x2_face.erase(faceEye_detector.x2_face.begin(),faceEye_detector.x2_face.end());
		faceEye_detector.y2_face.erase(faceEye_detector.y2_face.begin(),faceEye_detector.y2_face.end());
		
		if(faceEye_detector.detected_eyes.size() != 0) {
			// copy rectangle points of detected eyes
			memcpy(x1_eyes_out,&faceEye_detector.x1_eyes[0],faceEye_detector.detected_eyes.size()*sizeof(int));
			memcpy(y1_eyes_out,&faceEye_detector.y1_eyes[0],faceEye_detector.detected_eyes.size()*sizeof(int));
			memcpy(x2_eyes_out,&faceEye_detector.x2_eyes[0],faceEye_detector.detected_eyes.size()*sizeof(int));
			memcpy(y2_eyes_out,&faceEye_detector.y2_eyes[0],faceEye_detector.detected_eyes.size()*sizeof(int));
			
			faceEye_detector.x1_eyes.erase(faceEye_detector.x1_eyes.begin(),faceEye_detector.x1_eyes.end());
			faceEye_detector.y1_eyes.erase(faceEye_detector.y1_eyes.begin(),faceEye_detector.y1_eyes.end());
			faceEye_detector.x2_eyes.erase(faceEye_detector.x2_eyes.begin(),faceEye_detector.x2_eyes.end());
			faceEye_detector.y2_eyes.erase(faceEye_detector.y2_eyes.begin(),faceEye_detector.y2_eyes.end());
		}		

		return 0;
	}

	else return -1;
}