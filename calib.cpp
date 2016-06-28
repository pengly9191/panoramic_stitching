#include "FullViewStitch.h"

 //   http://blog.csdn.net/qq_15947787/article/details/51441031
string findFileName(int num);
int main(){

	vector<Point2f>corners;
	vector<vector<Point2f>>corners_Seq;
	vector<Mat>image_Seq;
		
	findCorner(corners,corners_Seq,image_Seq);


	return 1;
}

string findFileName(int num,string path,string pyte){
	string imageFileName;
	std::stringstream StrStm;
	StrStm << path;
	StrStm << i;
	StrStm >> imageFileName;
	imageFileName += pyte;
	return imageFileName;
}


void findCorner(vector<Point2f>&corners,vector<vector<Point2f>>&corners_Seq,vector<Mat>&image_Seq){

	cout<<"start find corners"<<endl; 
	int image_count = 25;
	Size board_size = Size(9,6);

	int successImageNum = 0;
	Mat image,imageGray;

	int count = 0;
	for(int i = 0 ;i<image_count;i++){
		image = imread(findFileName(i));
		cvtColor(image,imageGray,CV_RGB2GRAY);
		bool found = findChessboardCorners(image, board_size, corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ 
                CALIB_CB_FAST_CHECK);
		if (!found){
			cout <<"not found corners!"<< endl;
			continue;
		}else{

			cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
			Mat temp = image.clone();
			drawChessboardCorners( temp, board_size, Mat(corners), found );
			string imageFileName;
        	        std::stringstream StrStm;
        	        StrStm<<i+1;
        	        StrStm>>imageFileName;
        	        imageFileName += "_corner.jpg";
        	        imwrite(imageFileName,temp);
        	        cout<<"Frame corner#"<<i+1<<"...end"<<endl;

			 count = count + corners.size();
            		 successImageNum = successImageNum + 1;
            		 corners_Seq.push_back(corners);
		}
		 image_Seq.push_back(image);			
	}
	cout << " corners find success"<< endl;


}


void cameraCalib(){
	 cout<<"camera calib start "<<endl;  
	 Size square_size = Size(20,20);     
    	 vector<vector<Point3f>>  object_Points;  
		  Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0)); 
		  vector<int>  point_counts;   
		   for (int t = 0; t<successImageNum; t++)
    {
        vector<Point3f> tempPointSet;
        for (int i = 0; i<board_size.height; i++)
        {
            for (int j = 0; j<board_size.width; j++)
            {
                /* \u5047\u8bbe\u5b9a\u6807\u677f\u653e\u5728\u4e16\u754c\u5750\u6807\u7cfb\u4e2dz=0\u7684\u5e73\u9762\u4e0a */
                Point3f tempPoint;
                tempPoint.x = i*square_size.width;
                tempPoint.y = j*square_size.height;
                tempPoint.z = 0;
                tempPointSet.push_back(tempPoint);
            }
        }
        object_Points.push_back(tempPointSet);
    }

	 for (int i = 0; i< successImageNum; i++)
    {
        point_counts.push_back(board_size.width*board_size.height);
    }

	 // start calib
	 
	 Size image_size = image_Seq[0].size();
    cv::Matx33d intrinsic_matrix;  
    cv::Vec4d distortion_coeffs;    
    std::vector<cv::Vec3d> rotation_vectors;                      
    std::vector<cv::Vec3d> translation_vectors;                  
    int flags = 0;
    flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= cv::fisheye::CALIB_CHECK_COND;
    flags |= cv::fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
    cout<<"calib success \n";   	  
		 
double total_err = 0.0;
double err = 0.0; 

vector<Point2f>  image_points2;

 for (int i=0;  i<image_count;  i++) 
    {
        vector<Point3f> tempPointSet = object_Points[i];
        fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
        vector<Point2f> tempImagePoint = corners_Seq[i];
        Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
        Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
        for (size_t i = 0 ; i != tempImagePoint.size(); i++)
        {
            image_points2Mat.at<Vec2f>(0,i) = Vec2f(image_points2[i].x, image_points2[i].y);
            tempImagePointMat.at<Vec2f>(0,i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err/=  point_counts[i];   
        cout<<i+1<<"average error "<<err<<"pixel"<<endl;   
        fout<<i+1<<"average error"<<err<<"pixel"<<endl;   
    }   
    cout<<"total average error"<<total_err/image_count<<"pixel"<<endl;   

    



}




















