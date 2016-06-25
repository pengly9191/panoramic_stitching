#include "FullViewStitch.h"

#define ENABLE_LOG  1
#define  HAVE_OPENCV_XFEATURES2D  0
#define HAVE_OPENCV_CUDALEGACY 0

// define parament 
vector<String> img_names;
bool preview = false;   // if 1 the speed  is faster  to normal model  
bool try_cuda = false;  //
double work_megapix = 0.6;  //图像匹配的分辨率大小  resize image 
double seam_megapix = 0.1;   //拼接缝像素的大小  
double compose_megapix = 0.6;  //拼接分辨率 
float conf_thresh = 1.f;  //两幅图来自同一全景图的置信度
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;   
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;   //波形校验，水平  
bool save_graph = false;  
std::string save_graph_to; 
string warp_type = "spherical";     //弯曲类型
int expos_comp_type = Exposur  eCompensator::GAIN_BLOCKS;  //光照补偿方法，默认是gain_blocks  
float match_conf = 0.3f;    //特征点检测置信等级，最近邻匹配距离与次近邻匹配距离的比值，surf默认为0.65  
string seam_find_type = "gc_color";   
int blend_type = Blender::MULTI_BAND;   //融合方法，默认是多频段融合  
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;   /融合强度，0 - 100.默认是5
string result_name = "result_detail.jpg";
bool timelapse = false;
int timelapse_range = 5;


int main( )
{
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif	

	int argc = 8;  
	char* argv[ ] = {"1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg"};
	for (int i = 0; i < argc; ++i)  {
		img_names.push_back(argv[i]); 
	}	
	
 	
	
	// Check if have enough images
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
	    cout<<"Need more images"<<endl;
	    return -1;
	}


	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	
	//特征点检测以及对图像进行预处理（尺寸缩放），然后计算每幅图形的特征点，以及特征点描述子
	cout<<"Finding features..."<<endl;
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	Ptr<FeaturesFinder> finder;
	finder = makePtr<SurfFeaturesFinder>();   //Surf特征检测
	
	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i)
	{
	    Mat full_img = imread(img_names[i]);	
	    full_img_sizes[i] = full_img.size();

	    if (full_img.empty())
	    {
	        cout<<"Can't open image " << img_names[i]<<endl;
	        return -1;
	    }
	    if (work_megapix < 0)
	    {
	        img = full_img;
	        work_scale = 1;
	        is_work_scale_set = true;
	    }
	    else
	    {
	        if (!is_work_scale_set)
	        {
		    // 计算work_scale，取min值，将图像resize到面积在work_megapix*10^6以下
	            work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
	            is_work_scale_set = true;
	        }
	        resize(full_img, img, Size(), work_scale, work_scale);
	    }
	    if (!is_seam_scale_set)
	    {
	        // 计算seam_scale，将图像resize到面积在seam_megapix*10^6以下
	        seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
	        seam_work_aspect = seam_scale / work_scale;  //两次压缩比
	        is_seam_scale_set = true;
	    }

	    // 将原图像resize到work_scale尺寸下，计算图像特征点，以及计算特征点描述子，并将img_idx设置为i
	    (*finder)(img, features[i]);
	    features[i].img_idx = i;
		
	    cout << "Features in image #" << i+1 << ": " << features[i].keypoints.size()<< endl;
	 
	    //将原图像resize到seam_scale尺寸下  
	    resize(full_img, img, Size(), seam_scale, seam_scale);
	    images[i] = img.clone();
	}

	finder->collectGarbage();  // release
	full_img.release();
	img.release();

	cout <<"Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec"<< endl;

	cout <<"Pairwise matching"<< endl;
	// image match  
#if ENABLE_LOG
	t = getTickCount();
#endif

	// 图像两两匹配
	vector<MatchesInfo> pairwise_matches;
	if (!timelapse)
	{
		//使用最近邻和次近邻匹配，对任意两幅图进行特征点匹配  
	    BestOf2NearestMatcher matcher(try_cuda, match_conf);//最近邻和次近邻法
	    matcher(features, pairwise_matches);//对每两个图片进行匹配 
	    matcher.collectGarbage();
	}
	else
	{
	    BestOf2NearestRangeMatcher matcher(timelapse_range, try_cuda, match_conf);
	    matcher(features, pairwise_matches);
	    matcher.collectGarbage();
	}

	

	cout <<"Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec"<< endl;
	// Check if we should save matches graph

	// 输出匹配结果   Nm ：匹配的数量 Ni是内围数量在一个满足ransac单应矩阵  conf_thresh ：匹配信息的置信度 = 匹配信息的内围数/(8+0.3*匹配信息的匹配的大小);
	if (!save_graph)
	{
	    cout <<"Saving matches graph..."<< endl;
	    save_graph_to = "--save_graph";
	    ofstream f(save_graph_to.c_str());
	    f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	//将置信度高于门限的所有匹配合并到一个集合中  
        //只留下确定是来自同一全景图的图片 
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
	    img_names_subset.push_back(img_names[indices[i]]);
	    img_subset.push_back(images[indices[i]]);
	    full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
	    cout <<"Need more images"<< endl;
	    return -1;
	}
	
// estimate camera parament 
	//基于单应性的估计量
	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras; 
	if (!estimator(features, pairwise_matches, cameras))
	{
	    cout << "Homography estimation failed.\n";
	    return -1;
	}

	for (size_t i = 0; i < cameras.size(); ++i)
	{
	    Mat R;
	    cameras[i].R.convertTo(R, CV_32F);
	    cameras[i].R = R;
	   cout <<"Initial camera intrinsics parament #" << indices[i]+1 << ":\n" << cameras[i].K()<< endl;
	}

// using Bundle Adjustment way to calib the camera parament of all pictures 

	Ptr<detail::BundleAdjusterBase> adjuster;   //光束调整器参数 
	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();  //使用Bundle Adjustment（光束法平差）方法对所有图片进行相机参数校正  
	else
	{
	    cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
	    return -1;
	}
	adjuster->setConfThresh(conf_thresh);  //设置配置阈值
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))//进行矫正  
	{
	    cout << "Camera parameters adjusting failed.\n";
	    return -1;
	}

	// Find median focal length
	// 求出的焦距取中值和所有图片的焦距并构建camera参数，将矩阵写入camera  

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
	    LOGLN("Camera #" << indices[i]+1 << ":\n" << cameras[i].K());
	    focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
	    warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
	    warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	// wave correct  波形矫正  
	if (do_wave_correct)
	{
	    vector<Mat> rmats;
	    for (size_t i = 0; i < cameras.size(); ++i)
	        rmats.push_back(cameras[i].R.clone());
	    waveCorrect(rmats, wave_correct);
	    for (size_t i = 0; i < cameras.size(); ++i)
	        cameras[i].R = rmats[i];
	}

	LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);  //统一坐标后的顶点 
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<UMat> masks(num_images);  //融合掩码

	// Prepare images masks  准备图像融合掩码
	for (int i = 0; i < num_images; ++i)
	{
	    masks[i].create(images[i].size(), CV_8U);
	    masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks //弯曲图像和融合掩码  

	Ptr<WarperCreator> warper_creator;

#if 0
	
#ifdef HAVE_OPENCV_CUDAWARPING
	if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
	{
	    if (warp_type == "plane")
	        warper_creator = makePtr<cv::PlaneWarperGpu>();
	    else if (warp_type == "cylindrical")
	        warper_creator = makePtr<cv::CylindricalWarperGpu>();
	    else if (warp_type == "spherical")
	        warper_creator = makePtr<cv::SphericalWarperGpu>();
	}
	else
#endif
	{
	    if (warp_type == "plane")
	        warper_creator = makePtr<cv::PlaneWarper>();
	    else if (warp_type == "cylindrical")
	        warper_creator = makePtr<cv::CylindricalWarper>();
	    else if (warp_type == "spherical")
	        warper_creator = makePtr<cv::SphericalWarper>();
	    else if (warp_type == "fisheye")
	        warper_creator = makePtr<cv::FisheyeWarper>();
	    else if (warp_type == "stereographic")
	        warper_creator = makePtr<cv::StereographicWarper>();
	    else if (warp_type == "compressedPlaneA2B1")
	        warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
	    else if (warp_type == "compressedPlaneA1.5B1")
	        warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
	    else if (warp_type == "compressedPlanePortraitA2B1")
	        warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
	    else if (warp_type == "compressedPlanePortraitA1.5B1")
	        warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
	    else if (warp_type == "paniniA2B1")
	        warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
	    else if (warp_type == "paniniA1.5B1")
	        warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
	    else if (warp_type == "paniniPortraitA2B1")
	        warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
	    else if (warp_type == "paniniPortraitA1.5B1")
	        warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
	    else if (warp_type == "mercator")
	        warper_creator = makePtr<cv::MercatorWarper>();
	    else if (warp_type == "transverseMercator")
	        warper_creator = makePtr<cv::TransverseMercatorWarper>();
	}

	if (!warper_creator)
	{
	    cout << "Can't create the following warper '" << warp_type << "'\n";
	    return 1;
	}
#endif 

	warper_creator = new cv::SphericalWarper();  
//	warper_creator = makePtr<cv::SphericalWarper>();

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
	    Mat_<float> K;
	    cameras[i].K().convertTo(K, CV_32F);
	    float swa = (float)seam_work_aspect;
	    K(0,0) *= swa; K(0,2) *= swa;
	    K(1,1) *= swa; K(1,2) *= swa;

		//计算统一后坐标顶点  
	    corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
	    sizes[i] = images_warped[i].size();

		//弯曲当前图像
	    warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
	    images_warped[i].convertTo(images_warped_f[i], CV_32F);

//	LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

// exposure compensator 
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

// find seam 查找接缝  	
	Ptr<SeamFinder> seam_finder;
			
	 seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);	
	if (!seam_finder)
	{
	    cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
	    return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	//图像融合
	cout <<"Compositing...";
#if ENABLE_LOG
	t = getTickCount();
#endif

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;
	Ptr<Timelapser> timelapser;
	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
	    cout<<"Compositing image #" << indices[img_idx]+1<< endl;

	//由于以前进行处理的图片都是以work_scale进行缩放的，所以图像的内参  
        //corner（统一坐标后的顶点），mask（融合的掩码）都需要重新计算  
      
        // 读取图像和做必要的调整  

	    // Read image and resize it if necessary
	    Mat  full_img= imread(img_names[img_idx]);
				
	  
	    if (!is_compose_scale_set)
	    {
	        if (compose_megapix > 0)
	            compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
	        is_compose_scale_set = true;

	        // Compute relative scales
	        //compose_seam_aspect = compose_scale / seam_scale;
	        compose_work_aspect = compose_scale / work_scale;

	        // Update warped image scale 更新弯曲图像比例 
	        warped_image_scale *= static_cast<float>(compose_work_aspect);
	        warper = warper_creator->create(warped_image_scale);

	        // Update corners and sizes  
	        for (int i = 0; i < num_images; ++i)
	        {
	            // Update intrinsics  更新相机内参  
	            cameras[i].focal *= compose_work_aspect;
	            cameras[i].ppx *= compose_work_aspect;
	            cameras[i].ppy *= compose_work_aspect;

	            // Update corner and size
	            Size sz = full_img_sizes[i];
	            if (std::abs(compose_scale - 1) > 1e-1)
	            {
	                sz.width = cvRound(full_img_sizes[i].width * compose_scale);
	                sz.height = cvRound(full_img_sizes[i].height * compose_scale);
	            }

	            Mat K;
	            cameras[i].K().convertTo(K, CV_32F);
	            Rect roi = warper->warpRoi(sz, K, cameras[i].R);
	            corners[i] = roi.tl();
	            sizes[i] = roi.size();
	        }
	    }
	    if (abs(compose_scale - 1) > 1e-1)
	        resize(full_img, img, Size(), compose_scale, compose_scale);
	    else
	        img = full_img;
	    full_img.release();
	    Size img_size = img.size();

	    Mat K;
	    cameras[img_idx].K().convertTo(K, CV_32F);

	    // Warp the current image
	    warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

	    // Warp the current image mask
	    mask.create(img_size, CV_8U);
	    mask.setTo(Scalar::all(255));
	    warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

	    // Compensate exposure
	    compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

	    img_warped.convertTo(img_warped_s, CV_16S);
	    img_warped.release();
	    img.release();
	    mask.release();

	    dilate(masks_warped[img_idx], dilated_mask, Mat());
	    resize(dilated_mask, seam_mask, mask_warped.size());
	    mask_warped = seam_mask & mask_warped;

		 //初始化blender  
	    if (!blender)
	    {
	        blender = Blender::createDefault(blend_type, try_cuda);
		// computer result picture size
	        Size dst_sz = resultRoi(corners, sizes).size();
	        float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
	        if (blend_width < 1.f)
	            blender = Blender::createDefault(Blender::NO, try_cuda);
	        else if (blend_type == Blender::MULTI_BAND)
	        {
	            MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
	            mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
	            cout << "Multi-band blender, number of bands: " << mb->numBands()<<endl;
	        }
	        else if (blend_type == Blender::FEATHER)
	        {
	            FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
	            fb->setSharpness(1.f/blend_width);
	            LOGLN("Feather blender, sharpness: " << fb->sharpness());
	        }
		// sure last full view pic size according to coners and picture size根据corners顶点和图像的大小确定最终全景图的尺寸  
	        blender->prepare(corners, sizes);
	    }
	 
	    // Blend the current image	 	
	        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	   
	}
	
	    Mat result, result_mask;
	    blender->blend(result, result_mask);	   
	    imwrite(result_name, result);

	 int64 app_end_time = getTickCount();
	 cout << "app total time :"<< ((app_end_time -app_start_time) / getTickFrequency()) << " sec"<< endl;
	 		
	
	return 0;
	}
