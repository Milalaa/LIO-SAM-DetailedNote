/************************************************* 
GitHub: https://github.com/smilefacehh/LIO-SAM-DetailedNote
Author: lutao2014@163.com
Date: 2021-02-21 
--------------------------------------------------
功能简介:
    1、scan-to-map匹配：提取当前激光帧特征点（角点、平面点），局部关键帧map的特征点，执行scan-to-map迭代优化，更新当前帧位姿；
    2、关键帧因子图优化：关键帧加入因子图，添加激光里程计因子、GPS因子、闭环因子，执行因子图优化，更新所有关键帧位姿；
    3、闭环检测：在历史关键帧中找距离相近，时间相隔较远的帧设为匹配帧，匹配帧周围提取局部关键帧map，同样执行scan-to-map匹配，得到位姿变换，构建闭环因子数据，加入因子图优化。

订阅：
    1、lio_sam/feature/cloud_info：订阅当前激光帧点云信息，来自FeatureExtraction；
    2、gpsTopic：订阅GPS里程计；
    3、lio_loop/loop_closure_detection：订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上。


发布：
    1、发布历史关键帧里程计；
    2、发布局部关键帧map的特征点云；
    3、发布激光里程计，rviz中表现为坐标轴；
    4、发布激光里程计；
    5、发布激光里程计路径，rviz中表现为载体的运行轨迹；
    6、发布地图保存服务；
    7、发布闭环匹配关键帧局部map；
    8、发布当前关键帧经过闭环优化后的位姿变换之后的特征点云；
    9、发布闭环边，rviz中表现为闭环帧之间的连线；
    10、发布局部map的降采样平面点集合；
    11、发布历史帧（累加的）的角点、平面点降采样集合；
    12、发布当前帧原始点云配准之后的点云。
**************************************************/ 
#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose


/**
 * 6D位姿点云结构定义
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D     
    PCL_ADD_INTENSITY;  
    float roll;         
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   
} EIGEN_ALIGN16;                    

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;

    // 历史所有关键帧的角点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    // 历史所有关键帧的平面点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    // 历史关键帧位姿（位置）
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    // 历史关键帧位姿
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 当前激光帧角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; 
    // 当前激光帧平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; 
    // 当前激光帧角点集合，降采样，DS: DownSize
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; 
    // 当前激光帧平面点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; 

    // 当前帧与局部map匹配上了的角点、平面点，加入同一集合；后面是对应点的参数
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    // 当前帧与局部map匹配上了的角点、参数、标记
    std::vector<PointType> laserCloudOriCornerVec; 
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    // 当前帧与局部map匹配上了的平面点、参数、标记
    std::vector<PointType> laserCloudOriSurfVec; 
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    // 局部map的角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 局部map的平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 局部map的角点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 局部map的平面点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    // 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    cv::Mat matP;

    // 局部map角点数量
    int laserCloudCornerFromMapDSNum = 0;
    // 局部map平面点数量
    int laserCloudSurfFromMapDSNum = 0;
    // 当前激光帧角点数量
    int laserCloudCornerLastDSNum = 0;      
    // 当前激光帧面点数量
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    // 当前帧位姿
    Eigen::Affine3f transPointAssociateToMap;
    // 前一帧位姿
    Eigen::Affine3f incrementalOdometryAffineFront;
    // 当前帧位姿
    Eigen::Affine3f incrementalOdometryAffineBack;

    /**
     * 构造函数
    */
    mapOptimization()
    {
        // ISM2参数
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // 发布历史关键帧里程计
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        // 发布局部关键帧map的特征点云
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        // 发布激光里程计，rviz中表现为坐标轴
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        // 发布激光里程计，它与上面的激光里程计基本一样，只是roll、pitch用imu数据加权平均了一下，z做了限制
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        // 发布激光里程计路径，rviz中表现为载体的运行轨迹
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        // 订阅当前激光帧点云信息，来自featureExtraction
        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅GPS里程计
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布地图保存服务
        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        // 发布闭环匹配关键帧局部map
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        // 发布闭环边，rviz中表现为闭环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        // 发布局部map的降采样平面点集合
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        // 发布历史帧（累加的）的角点、平面点降采样集合
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        // 发布当前帧原始点云配准之后的点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    /**
     * 初始化
    */
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    /**
     * 订阅当前激光帧点云信息，来自featureExtraction
     * 1、当前帧位姿初始化
     *   1) 如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     *   2) 后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
     * 2、提取局部角点、平面点云集合，加入局部map
     *   1) 对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     *   2) 对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
     * 3、当前激光帧角点、平面点集合降采样
     * 4、scan-to-map优化当前帧位姿
     *   (1) 要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
     *   (2) 迭代30次（上限）优化
     *      1) 当前激光帧角点寻找局部map匹配点
     *          a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *          b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *      2) 当前激光帧平面点寻找局部map匹配点
     *          a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *          b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *      3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *      4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     *   (3)用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
     * 5、设置当前帧为关键帧并执行因子图优化
     *   1) 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
     *   2) 添加激光里程计因子、GPS因子、闭环因子
     *   3) 执行因子图优化
     *   4) 得到当前帧优化后位姿，位姿协方差
     *   5) 添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
     * 6、更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
     * 7、发布激光里程计
     * 8、发布里程计、点云、轨迹
    */
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // 当前激光帧时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // 提取当前激光帧角点、平面点集合
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);   //ros转成pcl格式
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        // mapping执行频率控制
        static double timeLastProcessing = -1;  
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)    //当前处理的时间距离上一帧要大于等于mappingProcessInterval（在params.yaml中设置为0.15s，即两帧处理一帧，这可以根据算力来决定大小）
        {
            timeLastProcessing = timeLaserInfoCur;

            // 当前帧位姿初始化；即更新当前匹配结果的初始位姿
            // 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
            // 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
            updateInitialGuess();   

            // 提取局部角点、平面点云集合，加入局部map；即提取当前帧相关的关键帧并且构建点云局部地图
            // 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
            // 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
            extractSurroundingKeyFrames();

            // 当前激光帧角点、平面点集合降采样；即对当前帧进行下采样
            downsampleCurrentScan();

            // scan-to-map优化当前帧位姿；即对点云配准进行优化问题构建求解
            // 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
            // 2、迭代30次（上限）优化
            //    1) 当前激光帧角点寻找局部map匹配点
            //       a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
            //       b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
            //    2) 当前激光帧平面点寻找局部map匹配点
            //       a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
            //       b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
            //    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
            //    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
            // 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
            scan2MapOptimization();

            // 设置当前帧为关键帧并执行因子图优化；即根据配准结果缺点是否是关键帧
            // 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            // 2、添加激光里程计因子、GPS因子、闭环因子
            // 3、执行因子图优化
            // 4、得到当前帧优化后位姿，位姿协方差
            // 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            saveKeyFramesAndFactor();

            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹；即调整全局轨迹
            correctPoses();

            // 发布激光里程计；即讲lidar里程计信息发送出去
            publishOdometry();

            // 发布里程计、点云、轨迹；即发送可视化点云
            // 1、发布历史关键帧位姿集合
            // 2、发布局部map的降采样平面点集合
            // 3、发布历史帧（累加的）的角点、平面点降采样集合
            // 4、发布里程计轨迹
            publishFrames();
        }
    }

    /**
     * 订阅GPS里程计，添加到GPS里程计队列
    */
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    /**
     * 激光坐标系下的激光点，通过激光帧位姿，变换到世界坐标系下
    */
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    /**
     * 对点云cloudIn进行变换transformIn，返回结果点云
    */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);
        // 使用openmp进行并行加速
        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            // 每个点都施加Rx+t这样一个过程
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    /**
     * 位姿格式变换
    */
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    /**
     * 位姿格式变换
    */
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    /**
     * Eigen格式的位姿变换
    */
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    /**
     * Eigen格式的位姿变换
    */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    /**
     * 位姿格式变换
    */
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    












    /**
     * 保存全局关键帧特征点集合
    */
    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // 这个代码太坑了！！注释掉
      // int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      // unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // 保存历史关键帧位姿
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // 提取历史关键帧角点、平面点集合
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // 降采样
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // 降采样
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // 保存到一起，全局关键帧特征点集合
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    /**
     * 展示线程
     * 1、发布局部关键帧map的特征点云
     * 2、保存全局关键帧特征点集合
    */
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);    // 更新频率设置为0.2hz
        while (ros::ok()){
            rate.sleep();
            // 发布局部关键帧map的特征点云
            publishGlobalMap();
        }
        // 当ros被杀死后，执行保存地图功能
        if (savePCD == false)
            return;

        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        // 保存全局关键帧特征点集合
        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    /**
     * 发布可视化全局地图
    */
    void publishGlobalMap()
    {
        // 如果没有订阅者就不发布，节省系统负载
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;
        // 没有关键帧自然也没有全局地图了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);    // 把所有关键帧送入kdtree
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);  // 寻找自信关键帧一定范围内的其他关键帧
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)   // 把这些找到的关键帧的位姿保存起来
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // 降采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            // 距离过大
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 将每一帧的点云通过位姿转到世界坐标系下
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // 降采样，发布
        // 转换后的点云也进行一个下采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // 最终发布出去
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }











    /**
     * 回环线程
     * 1、闭环scan-to-map，icp优化位姿
     *   1) 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     *   2) 提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
     *   3) 执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
     * 2、rviz展示闭环边
    */
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false) // 如果不需要进行回环检测，那么就退出这个线程
            return;

        ros::Rate rate(loopClosureFrequency);   // 设置回环检测的频率；loopClosureFrequency在param.yaml中也有设置为1hz（1秒执行一次）
        while (ros::ok())   // 死循环
        {
            rate.sleep();   // 执行完一次就必须sleep一段时间，否则该线程的cpu占用会非常高；同ros::Rate rate(loopClosureFrequency)中时间，这里为1s
            // 闭环scan-to-map，icp优化位姿
            // 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
            // 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
            // 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
            // 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
            performLoopClosure();   // 执行回环检测
            // rviz展示闭环边
            visualizeLoopClosure();
        }
    }

    /**
     * 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上
    */
    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    /**
     * 闭环scan-to-map，icp优化位姿
     * 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
     * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
     * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
    */
    void performLoopClosure()
    {
        // 如果没有关键帧，就没法进行回环检测了
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock(); //线程锁
        // 把存储关键帧的位姿的点云copy出来，避免线程冲突
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;   // 关键帧的xyz，不带姿态 
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;   // 关键帧的xyz+姿态
        mtx.unlock();

        // 当前关键帧索引，候选闭环匹配帧索引
        int loopKeyCur;
        int loopKeyPre;
        // not-used
        // 首先看一下外部通知的回环消息（手动的告知第几帧和第几帧之间形成回环）
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)   // loopKeyCur当前帧id，loopKeyPre回环帧id
                return;

        // 检测出回环之后开始计算两帧相对位姿变换（icp），点云配准scan-to-map
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());    // 空的点云
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 提取当前关键帧特征点集合，降采样；点云1
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);    // 当前帧（scan）
            // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样；点云2
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); // 历史帧（map），historyKeyframeSearchNum历史帧搜索范围（param.yaml中设置为25）
            // 如果点云数目较少，返回
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            // 发布闭环匹配关键帧局部map
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                // 把局部地图发布出去供rviz可视化使用
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // 使用简单的ico进行帧到局部地图的配准；ICP参数设置
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);    // 最大相关距离；historyKeyframeSearchRadius=15
        icp.setMaximumIterations(100);  // 最大优化次数
        icp.setTransformationEpsilon(1e-6); // 单次变化范围
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // scan-to-map，调用icp匹配
        // 设置两个点云
        icp.setInputSource(cureKeyframeCloud);  // 当前（scan）
        icp.setInputTarget(prevKeyframeCloud);  // 之前（map）
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);  // 执行点云配准

        // 检查icp是否收敛且得分是否满足要求
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云；可视化
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        
        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // 闭环优化后当前帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);    // 将闭环优化后当前帧位姿转成平移和欧拉角
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z)); // poseFrom是闭环优化后当前帧位姿
        // 闭环匹配帧的位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);   // poseTo是闭环关键帧（前）的位姿
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();   // 使用icp的得分作为她们约束的噪声项
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // 添加闭环因子需要的数据
        // 将两帧索引，两帧相对位姿和噪声作为回环约束送入队列（gtsam，主进程使用队列进行优化）
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();
        // 保存已存在的约束对
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     * 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 当前关键帧帧
        // 检测最新帧是否和其他帧形成回环，所以后面一帧的id就是最后一个关键帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;  // （当前帧）最新帧的索引，size-1；copy_cloudKeyPose3D历史关键帧
        int loopKeyPre = -1;    // 之前帧（回环帧）

        // 当前帧已经添加过闭环对应关系，不再继续添加
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); // 把只包含关键帧位移信息的点云填充kdtree
        // copy_cloudKeyPoses3D->back()最新关键帧，historyKeyframeSearchRadius为半径搜索关键帧（param.yaml中设置为15），存储于pointSearchIndLoop, pointSearchSqDisLoop
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);    
        // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // 必须满足时间上超过一定阈值historyKeyframeSearchTimeDiff（param.yaml设置为30），才认为是一个有效的回环
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                // 此时就退出了
                loopKeyPre = id;
                break;
            }
        }
        // 如果没有找到回环或者回环找到自己身上去了，就认为是此次回环寻找失败
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * not-used, 来自外部闭环检测程序提供的闭环匹配索引对（手动的告知第几帧和第几帧之间会形成回环）
    */
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
    */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        // searchNum是搜索范围
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;  // searchNum是相对的索引差，和当前的key补上得出绝对
            if (keyNear < 0 || keyNear >= cloudSize )   // 如果超出范围就不行
                continue;
            // 否则把对应角点和面点的点云转到世界坐标系下去，并且加到同一点云去
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }
        // 如果没有有效点云就算了
        if (nearKeyframes->empty())
            return;

        // 把点云下采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    /**
     * rviz展示闭环边，将已有的回环约束可视化出来
    */
    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty()) //  如果没有回环就算了
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // 闭环顶点，回环约束的两帧作为node添加
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // 闭环边，两帧之间约束作为edge添加
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        // 遍历所有闭环约束
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            // 添加node和edge
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        // 最后发不出去供可视化使用
        pubLoopConstraintEdge.publish(markerArray);
    }







    


    /**
     * 当前帧位姿初始化(作为基于优化的点云匹配，初始值是非常重要的，一个好的初始值会帮助优化问题快速收敛且避免局部最优解的情况)
     * 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     * 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
    */
    void updateInitialGuess()
    {
        // transformTobeMapped是前一帧优化后的最佳位姿，注：这里指lidar的位姿，后面都简写成位姿
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);   //转换

        // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的位姿（旋转部分）
        static Eigen::Affine3f lastImuTransformation;
        
        // initialization 初始化
        // 如果关键帧集合为空，继续进行初始化（收到是第一帧点云）
        if (cloudKeyPoses3D->points.empty())
        {
            // 当前帧位姿的旋转部分，用激光帧信息中的RPY（来自imu原始数据，）初始化；初始位姿就由（imu）磁力计提供（磁力计---？？？）
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;
            // 无论是vio还是lio，，系统的不可观测都是四自由度，平移+yaw角，这里虽然由磁力计将yaw对齐，但是也可以考虑不使用yaw
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;
            // 保存（imu）磁力计得到的位姿，平移xyz设置为0
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); 
            return;
        }
        
        // use imu pre-integration estimation for pose guess
        // 用当前帧和前一帧对应的imu里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        // 如果有预积分节点提供的里程计
        if (cloudInfo.odomAvailable == true)
        {
            // 将提供的初值转成eigen的数据结构保存下来
            // 当前帧的初始估计位姿（来自imu里程计），后面用来计算增量位姿变换
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            // 这个标志位表示是否收到过第一帧预积分里程计信息
            if (lastImuPreTransAvailable == false)
            {
                // 赋值给前一帧
                lastImuPreTransformation = transBack;   // 将当前里程计结果记录下来
                lastImuPreTransAvailable = true;        // 收到第一个里程计数据以后这个标志位就是true
            } else {
                // 当前帧相对于前一帧的位姿变换，imu里程计计算得到
                // 计算上一个里程计的结果和当前里程计结果之间的delta pose
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                // 前一帧的位姿
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 当前帧的位姿
                // 将这个增量加入到上一帧最佳位姿上去，就是当前帧位姿的一个先验估计位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 更新当前帧位姿transformTobeMapped
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                // 同理，把当前帧的值保存下来
                // 赋值给前一帧
                lastImuPreTransformation = transBack;
                // 虽然有里程计信息，仍然需要把imu磁力计得到的旋转记录下来
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess（only rotation）
        // 如果没有里程计信息，就是用imu的旋转信息来更新，因为单纯使用imu无法得到靠谱的平移信息，因此平移之间设置为0
        // 只在第一帧调用（注意上面的return），用imu数据初始化当前帧位姿，仅初始化旋转部分
        if (cloudInfo.imuAvailable == true)
        {
            // 当前帧的姿态角（来自原始imu数据）
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            // 当前帧相对于前一帧的姿态变换
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            // 前一帧的位姿
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            // 当前帧的位姿
            Eigen::Affine3f transFinal = transTobe * transIncre;
            // 更新当前帧位姿transformTobeMapped
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    /**
     * not-used
    */
    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        // 
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
        extractCloud(cloudToExtract);
    }

    /**
     * 提取局部角点、平面点云集合，加入局部map
     * 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     * 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
    */
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;    //保存kdtree提取出来的元素的索引
        std::vector<float> pointSearchSqDis;    // 保存距离查询位置的距离的数组

        // kdtree的输入，全局关键帧位姿集合（历史所有关键帧集合）
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); 
        // 对最近的一帧关键帧，在半径区域内搜索空间区域上相邻的关键帧集合；surroundingKeyframeSearchRadius在.param中设置
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        // 遍历搜索结果，pointSearchInd存的是结果在cloudKeyPoses3D下面的索引
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            // 加入相邻关键帧位姿集合中
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 避免关键帧过多，降采样一下
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // 加入时间上相邻的一些关键帧，比如当载体在原地转圈，这些帧加进来是合理的
        int numPoses = cloudKeyPoses3D->size();
        // 刚刚是提取了一些空间上比较近的关键帧，然后再提取一些时间上比较近的关键帧
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)  //  最近十秒的关键帧也保存下来
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
        extractCloud(surroundingKeyPosesDS);
    }

    /**
     * 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
    */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // 分别存储角点和面点相关的局部地图
        // 相邻关键帧集合对应的角点、平面点，加入到局部map中；注：称之为局部map，后面进行scan-to-map匹配，所用到的map就是这里的相邻关键帧对应点云集合
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 遍历当前帧（实际是取最近的一个关键帧来找它相邻的关键帧集合）时空维度上相邻的关键帧集合
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 距离超过阈值，丢弃
            // 简单校验下关键帧距离不能太远，这个实际上不太会触发
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            // 相邻关键帧索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 如果这个关键帧对应的点云信息已经存储在一个地图容器里（为减少同一个点云转到世界坐标系下重复操作运算量）
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // 直接从容器中取出来加到局部地图中
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // 相邻关键帧对应的角点、平面点云，通过6D位姿变换到世界坐标系下
                // 如果这个点云没有实现存取，那就通过该帧对应的位姿，把该帧点云从当前帧的位姿转到世界坐标系下
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                // 点云转换后加到局部map
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                // 把转换后的面点和角点存进这个容器中，方便后续直接加入点云地图，避免点云转换的操作，节约时间
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // 将提取的关键帧的点云转到世界坐标系下后，避免点云过度密集，因此对面点和角点的局部地图做一个下采样的过程
        // 降采样局部角点map，在params.yaml文件中设置mappingCornerLeafsize=0.2
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // 降采样局部平面点map，在params.yaml文件中设置mappingSurfLeafsize=0.4，比起角点面点更为稀疏
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // 如果这个局部地图容器太大，就清空一下内存
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    /**
     * 提取局部角点、平面点云集合，加入局部map
     * 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     * 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
    */
    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true) // 如果当前没有关键帧，就return了
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        // 提取局部角点、平面点云集合，加入局部map
        // 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
        // 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
        extractNearby();
    }

    /**
     * 当前激光帧角点、平面点集合降采样，也是为了减少计算量（scan-to-map，map要下采样，scan同理也要）
    */
    void downsampleCurrentScan()
    {
        // 当前激光帧角点集合降采样
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        // 当前激光帧平面点集合降采样
        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    /**
     * 更新当前帧位姿
    */
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped); // transformTobeMapped是数组，把他转成eigen的一个对象更好操作
    }

    /**
     * 当前激光帧角点寻找局部map匹配点；边缘点的优化
     * 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     * 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
    */
    void cornerOptimization()
    {
        // 更新当前帧位姿
        updatePointAssociateToMap();

        // 使用openmp并行加速（如果每个循环都是相互独立的就可以使用）
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历当前帧角点集合
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;    // pointSel 搜索点；pointSearchInd 找到的5个点的索引；pointSearchsqDis 5个点的距离（pcl自动从小到大排序）；
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 角点（坐标还是lidar系）
            pointOri = laserCloudCornerLastDS->points[i];
            // 根据当前帧位姿，变换到世界坐标系（map系）下
            pointAssociateToMap(&pointOri, &pointSel);
            // 在局部角点map中查找当前角点相邻的5个角点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
            
            // 要求距离都小于1m；计算找到的点中距离当前点最远的点，如果距离太大说明这个约束不可信，就跳过
            if (pointSearchSqDis[4] < 1.0) {
                // 计算5个点的均值坐标，记为中心点
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 计算协方差
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    // 计算点与中心点之间的距离
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 构建协方差矩阵（是个对称的）
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 特征值分解（证明这些边缘点是不是直线）
                cv::eigen(matA1, matD1, matV1); // matD1：matA1的线特征值；matV1:matA1的线特征向量

                // 如果最大的特征值相比次大特征值，大3倍，认为构成了线，角点是合格的
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    // 当前帧角点坐标（map系下）
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 局部map对应中心角点，沿着特征向量（直线方向）方向，前后各取一个点
                    // 特征向量对应的就是直线的方向向量；通过点的均值往两边拓展，构成一个线的两个端点
                    // matV1.at<float>(0, 0)即最大特征值matD1.at<float>(0, 0)对应的特征向量
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);
                    
                    // 下面是计算点到线的残差和垂线方向（即雅克比方向，梯度下降方向）
                    // area_012，也就是三个点组成的三角形面积*2，叉积的模|axb|=a*b*sin(theta)
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    
                    // line_12，底边边长
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                    
                    // 两次叉积，得到点到直线的垂线段单位向量，x分量，下面同理
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // 三角形的高，也就是点到直线距离
                    float ld2 = a012 / l12;

                    // 一个简单的核函数，（残差）距离越大，s越小，是个距离惩罚因子（权重）
                    float s = 1 - 0.9 * fabs(ld2);

                    // 点到直线的垂线段单位向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // 点到直线距离
                    coeff.intensity = s * ld2;
                    // 如果残差小于10cm，就认为是一个有效的约束；匹配
                    if (s > 0.1) {
                        // 当前激光帧角点，加入匹配集合中
                        laserCloudOriCornerVec[i] = pointOri;
                        // 角点的参数
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 当前激光帧平面点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     * 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
    */
    void surfOptimization()
    {
        // 更新当前帧位姿
        updatePointAssociateToMap();

         // 遍历当前帧平面点集合
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            
            // 平面点（坐标还是lidar系）
            pointOri = laserCloudSurfLastDS->points[i];
            // 根据当前帧位姿，变换到世界坐标系（map系）下
            pointAssociateToMap(&pointOri, &pointSel); 
            // 在局部平面点map中查找当前平面点相邻的5个平面点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;   // matA0:5*3
            Eigen::Matrix<float, 5, 1> matB0;   // matB0:5*1
            Eigen::Vector3f matX0;
            // 平面方程Ax+By+Cz+1=0（这里不是用特征值分解的方法，而是用差变方程）d=1
            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // 要求距离都小于1m
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 假设平面方程为ax+by+cz+1=0，这里就是求方程的系数abc，d=1
                // 求解Ax=b这个超定方程
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // 平面方程Ax+By+Cz+1=0的系数，也是法向量的分量
                // 求出来的x就是这个平面的法向量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // 单位法向量
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps; // 归一化，将法向量模长统一为1（为了方便计算点到平面的距离）

                // 检查平面是否合格，如果5个点中有点到平面的距离超过0.2m，那么认为这些点太分散了，不构成平面
                // 每个点代入平面方程，计算点到平面的距离，如果距离大于0.2m，认为这个平面曲率偏大，就是无效的平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 如果平面合格
                if (planeValid) {
                    // 当前激光帧点到平面距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));  // 分母：当前点投到世界坐标系的模长再开根号

                    // 点到平面垂线单位法向量（雅可比方向，其实等价于平面法向量）
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    // 点到平面距离
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {  //匹配
                        // 当前激光帧平面点，加入匹配集合中
                        laserCloudOriSurfVec[i] = pointOri;
                        // 平面的参数
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合；即将角点约束和面点约束统一到一起
    */
    void combineOptimizationCoeffs()
    {
        // 遍历当前帧角点集合，提取出与局部map匹配上了的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){    // 只有标志位为true的时候才是有效约束
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // 遍历当前帧平面点集合，提取出与局部map匹配上了的平面点
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // 清空标记位
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    /**
     * scan-to-map优化
     * 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     * 公式推导：todo
     * 实际上是高斯牛顿，并不涉及LM（同loam）
    */
    bool LMOptimization(int iterCount)
    {
        // 原始的loam代码是将lidar坐标系转到相机坐标系，这里把原先loam中的代码拷贝了过来，但是为了坐标系的统一，就先转到相机坐标系优化，然后结果转回lidar坐标系
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // 当前帧匹配特征点数太少
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // 遍历匹配特征点，构建Jacobian矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            // 首先将当前点以及点到线（面）的单位向量转到坐标系
            // lidar -> camera todo
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 相机系下的旋转顺序是Y-X-Z对应lidar系下Z-Y-X
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            // 这里就是把camera转到lidar了
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            // 点到直线距离、平面距离，作为观测值
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        // 构成JTJ以及-JTe矩阵
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 求解增量；J^T·J·delta_x = -J^T·f 高斯牛顿
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);
 
        // 首次迭代，检查近似Hessian矩阵（J^T·J）是否退化，或者称为奇异，行列式值=0 todo
        if (iterCount == 0) {
            // 检查一下是否有退化的情况
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            // 对JTJ进行特征值分解
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                // 特征值从小到大遍历，如果小于阈值就认为退化
                if (matE.at<float>(0, i) < eignThre[i]) {
                    // 对应的特征向量全部置0
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }
        // 如果发生退化，就对增量进行修改，退化方向不更新
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 增量更新，更新当前位姿 x = x + delta_x
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // 旋转和平移增量足够小，认为优化问题收敛了
        // delta_x很小，认为收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; 
        }
        // 否则继续优化
        return false; 
    }

    /**
     * scan-to-map优化当前帧位姿
     * 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
     * 2、迭代30次（上限）优化
     *   1) 当前激光帧角点寻找局部map匹配点
     *      a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *      b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *   2) 当前激光帧平面点寻找局部map匹配点
     *      a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *      b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *   3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *   4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     * 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    */
    void scan2MapOptimization()
    {
        // 如果没有关键帧，那也没办法做当前帧到局部地图的匹配
        if (cloudKeyPoses3D->points.empty())
            return;

        // 判断当前激光帧的角点、平面点数量是否足够多；edgeFeatureMinValidNum&surfFeatureMinValidNum在param.yaml文件中设定
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 分别把角点面点局部地图构建kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 迭代30次（这里是手写优化器，继承了loam、lego_loam写法，是高斯牛顿的方法；aloam是通过ceres）
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                // 每次迭代清空特征点集合
                laserCloudOri->clear();
                coeffSel->clear();

                // 当前激光帧角点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
                // 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
                cornerOptimization();

                // 当前激光帧平面点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
                // 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
                surfOptimization();

                // 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                combineOptimizationCoeffs();

                // scan-to-map优化
                // 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                if (LMOptimization(iterCount) == true)
                    break;              
            }
            // 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    /**
     * 把结果和imu进行一些融合。用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    */
    void transformUpdate()
    {
        // 可以获取九轴imu的世界系下的姿态
        if (cloudInfo.imuAvailable == true)
        {
            // 因为roll和pitch原则上全程可观，因此这里把lidar推算出来的姿态和磁力计结果做一个加权平均
            // 首先判断车翻了没（判断俯仰角），车翻了做slam就无意义了，当然手持设备可以pitch很大，这里主要避免插值产生的奇异
            // 俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;    //  imuPPYWeight从Param.yaml中获得，为0.01，说明lidar的权重为99%
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);   // lidar匹配获得的roll角转成四元数
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);  //imu获得的roll角
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);   // 使用四元数球面插值
                transformTobeMapped[0] = rollMid;   // 插值结果作为roll的最终结果

                // pitch角同理
                // pitch角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // 更新当前帧位姿的roll, pitch, z坐标；因为是小车，roll、pitch是相对稳定的，不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
        // 主要针对室内2d场景下，已知先验可以加上这些约束
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        // 当前帧位姿
        // 最终结果也可以转成eigen结构
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    /**
     * 值约束
    */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    /**
     * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    */
    bool saveFrame()
    {
        // 如果没有关键帧，那直接认为是关键帧
        if (cloudKeyPoses3D->points.empty())
            return true;

        // 取出前一关键帧位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());   // cloudKeyPose6D所有关键帧的信息，back取最后一个元素
        // 当前帧的位姿transformTobeMapped转成eigen形式
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 计算两个位姿之间的delta pose
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        // 转成平移角和欧拉角的形式
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 旋转和平移量都较小，当前帧不设为关键帧，surroundingkeyframeAddingAngleThreshold阈值在params.yaml中设置为0.2rad，surroundingkeyframeAddingDistThreshold为1m
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    /**
     * 添加激光里程计因子；odom因子
    */
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())    // 如果是第一帧关键帧
        {
            // 第一帧初始化先验因子
            // 置信度就设置差一点，尤其是不可观的平移和yaw角（yaw角不一定能和磁力计对其，pich和raw和重力对齐；平移和yaw不可观符合四自由度不可观）
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 增加先验约束，对第0个节点增加约束
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 加入节点信息；变量节点设置初始值
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            // 如果不是第一帧，就增加帧间约束
            // 这时帧间约束置信度就设置高一些
            // 添加激光里程计因子
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 转成gtsam格式
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 帧间约束；参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差（置信度）
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 加入节点信息；变量节点设置初始值
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    /**
     * 添加GPS因子
    */
    void addGPSFactor()
    {
        // 如果没有gps信息就算了
        if (gpsQueue.empty())
            return;

        // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            // 第一个关键帧和最后一个关键帧相差相近也算了，要么刚起步，要么会触发回环
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // gtsam反馈当前x，y的置信度，位姿协方差（置信度高）很小，没必要加入GPS数据进行校正；poseCovThreshold在params.yaml中设置为25
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // 删除当前帧0.2s之前的里程计
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                gpsQueue.pop_front();
            }
            // 超过当前帧0.2s之后，退出，等等lidar再计算
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                break;
            }
            else
            {
                // 说明这个gps时间上距离当前帧已经比较近了，那就把这个数据取出来
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS噪声协方差太大（置信度太低），不能用
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold) // gpsCovThreshold在params.yaml中设置为2m
                    continue;

                // 取出GPS里程计位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)   // 通常gps的z没有x y准，因此这里可以不使用z值（useGpsElevation=false在params.yaml中设置）；lidar在z轴漂移比较严重，如果gps信号比较好，z漂移不怎么严重的话，可以考虑用起来
                {
                    gps_z = transformTobeMapped[5]; //没有用GPS的z值，用的里程计估计出来的z
                    noise_z = 0.01;
                }

                // (0,0,0)无效数据
                // gps的x或者y太小说明还没有初始化好
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // 每隔5m添加一个GPS里程计
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0) // 加入gps观测不宜太频繁，相邻不超过5m
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                // 添加GPS因子
                gtsam::Vector Vector3(3);
                // gps的置信度，标准差设置为最小1m，也就是不会特别信任gps信号
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);  
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                // 调用gtsam中集成的gps约束
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);    // cloudKeyPoses3D->size()当前帧还没有加进来，就还是size
                // 加入gps之后等同于回环，需要触发较多的isam update
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    /**
     * 添加闭环因子
    */
    void addLoopFactor()
    {
        // 有一个专门的回环检测线程会检测回环，检测到就会给这个队列塞入回环结果
        if (loopIndexQueue.empty())
            return;

        // 把队列里所有的回环约束都添加进来
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            // 闭环边对应两帧的索引
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 闭环边的位姿变换，帧间约束
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // 回环的置信度就是icp的得分
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            //加入约束
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }
        // 清空回环相关队列
        loopIndexQueue.clear(); 
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        // 标志位设置为true
        aLoopIsClosed = true;
    }

    /**
     * 设置当前帧为关键帧并执行因子图优化
     * 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
     * 2、添加激光里程计因子、GPS因子、闭环因子
     * 3、执行因子图优化
     * 4、得到当前帧优化后位姿，位姿协方差
     * 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
    */
    void saveKeyFramesAndFactor()
    {
        // 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        if (saveFrame() == false)
            return;

        // 激光里程计因子；odom因子
        addOdomFactor();

        // GPS因子
        addGPSFactor();

        // 闭环因子
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");
        
        // 所有因子加完了，就调用isam接口更新图模型
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        // 如果加入了gps的约束或者回环约束，isam需要进行更多次的优化
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        // update之后要清空一下保存的因子图（约束和节点），注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();
        // 下面保存关键帧信息
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // 优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        // 取出优化后的最新关键帧位姿
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 平移信息取出来保存进cloudKeyPoses3D这个结构中，其中索引作为intensity值
        // cloudKeyPoses3D加入当前帧位姿
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); 
        cloudKeyPoses3D->push_back(thisPose3D);

        // cloudKeyPoses6D加入当前帧位姿
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; 
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 保存当前位姿的置信度（位姿协方差）
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // 将优化后的位姿更新到transformTobeMapped数组中更新当前帧位姿，作为当前最佳估计值
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // 当前帧激光角点、平面点分别拷贝一下，降采样集合
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // 关键帧的点云保存下来
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // 根据当前最新位姿更新rviz可视化
        updatePath(thisPose6D);
    }

    /**
     * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
    */
    void correctPoses()
    {
        // 没有关键帧，自然也没有什么意义
        if (cloudKeyPoses3D->points.empty())
            return;

        // 只有回环以及gps信息这些会触发全局调整信息才回触发
        if (aLoopIsClosed == true)
        {
            // 很多位姿会发生变化，因子之前的容器内转到世界坐标系下的很多点云就需要调整，因此这里索性清空
            laserCloudMapContainer.clear();
            // 清空里程计轨迹path
            globalPath.poses.clear();
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 更新所有关键帧的位姿
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 更新里程计轨迹path
                updatePath(cloudKeyPoses6D->points[i]);
            }
            // 标志位置位
            aLoopIsClosed = false;
        }
    }

    /**
     * 更新里程计轨迹
    */
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * 发布激光里程计
    */
    void publishOdometry()
    {
        // 发布激光里程计，odom等价map；发送当前帧的位姿
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        pubLaserOdometryGlobal.publish(laserOdometryROS);

        // 发布TF，odom->lidar；发生lidar在odom坐标系下的tf
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        // 发送增量位姿变化
        // 这里主要用于给imu预积分模块使用，需要里程计是平滑的
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental;
        static Eigen::Affine3f increOdomAffine; 
        // 该标志位处理一次后始终位true
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;    // 记录当前位姿
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            // 当前帧与前一帧之间的位姿变换（scan matching之后，而不是根据回环或者gps调整之后的位姿）之间的位姿增量
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            // 还是当前帧位姿（打印一下数据，可以看到与激光里程计的位姿transformTobeMapped是一样的）
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;    // 分解成欧拉角和平移向量
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            // 如果有imu信号，同样对roll和pitch做插值
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // roll姿态角加权平均
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // pitch姿态角加权平均
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    /**
     * 发布里程计、点云、轨迹
     * 1、发布历史关键帧位姿集合
     * 2、发布局部map的降采样平面点集合
     * 3、发布历史帧（累加的）的角点、平面点降采样集合
     * 4、发布里程计轨迹
    */
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // 发布历史关键帧位姿集合
        publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // 发布局部map的降采样平面点集合；发送周围局部地图点云信息
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        
        // 发布历史帧（累加的）的角点、平面点降采样集合
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 把当前帧点云转换到世界坐标系下去
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            // 发送当前点云
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // 发布当前帧原始点云配准之后的点云
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            // 发送原始点云
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // 发布里程计轨迹path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");   // ros初始化

    mapOptimization MO; // 定义mapOptimization类

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");   // 打印消息
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);   //  回环检测的线程，专门进行回环检测
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);    // 可视化进程，负责rviz可视化的发布

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
