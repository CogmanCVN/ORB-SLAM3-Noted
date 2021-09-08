/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Atlas.h"
#include "ORBVocabulary.h"
#include "Tracking.h"
#include "Config.h"

#include "KeyFrameDatabase.h"

#include <boost/algorithm/string.hpp>
#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM3
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;
class Map;


class LoopClosing
{
public:

    typedef pair<set<KeyFrame*>,int> ConsistentGroup;    
    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3> > > KeyFrameAndPose;

public:

    LoopClosing(Atlas* pAtlas, KeyFrameDatabase* pDB, ORBVocabulary* pVoc,const bool bFixScale);

    void SetTracker(Tracking* pTracker);

    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();
    void RequestResetActiveMap(Map* pMap);

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF);

    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }
    bool isFinishedGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish();

    bool isFinished();

    Viewer* mpViewer;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef REGISTER_TIMES
    double timeDetectBoW;

    std::vector<double> vTimeBoW_ms;
    std::vector<double> vTimeSE3_ms;
    std::vector<double> vTimePRTotal_ms;

    std::vector<double> vTimeLoopFusion_ms;
    std::vector<double> vTimeLoopEssent_ms;
    std::vector<double> vTimeLoopTotal_ms;

    std::vector<double> vTimeMergeFusion_ms;
    std::vector<double> vTimeMergeBA_ms;
    std::vector<double> vTimeMergeTotal_ms;

    std::vector<double> vTimeFullGBA_ms;
    std::vector<double> vTimeMapUpdate_ms;
    std::vector<double> vTimeGBATotal_ms;
#endif

protected:

    bool CheckNewKeyFrames();

    //Methods to implement the new place recognition algorithm
    bool NewDetectCommonRegions();
    bool DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                        std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
    bool DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                     int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
    bool DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                            std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs);
    int FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                vector<MapPoint*> &vpMatchedMapPoints);


    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints);
    void SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints);

    void CorrectLoop();

    void MergeLocal();
    void MergeLocal2();

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetActiveMapRequested;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;
    Tracking* mpTracker;

    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue; // 回环线程中关键帧队列，关键帧是在LocalMapping线程中插入的

    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    float mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame* mpCurrentKF;
    KeyFrame* mpLastCurrentKF;
    KeyFrame* mpMatchedKF;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*> mvpCurrentConnectedKFs; // mpCurrentKF及其共视KF的关键帧数组
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    std::vector<MapPoint*> mvpLoopMapPoints;
    cv::Mat mScw;
    g2o::Sim3 mg2oScw;

    //-------
    Map* mpLastMap;

    bool mbLoopDetected; // true: 探测到回环；false: 不存在回环
    int mnLoopNumCoincidences; // 帧间存在共视情况的次数, 超过3次，则认为存在构成回环， mbLoopDetected = true
    int mnLoopNumNotFound;
    KeyFrame* mpLoopLastCurrentKF; // 上一次执行闭环检测的关键帧（上一次执行闭环检测是在当前帧为mpLoopLastCurrentKF时）
    g2o::Sim3 mg2oLoopSlw; // mg2oLoopSlw是根据候选帧的位姿和候选帧到当前帧的Sim3相对位姿，推算得到的当前帧在闭环时的目标位姿，简称为新位姿
    g2o::Sim3 mg2oLoopScw; // 当前帧闭环计算后的目标位姿，简称为新位姿
    KeyFrame* mpLoopMatchedKF; // 闭环检测时的最佳候选帧
    std::vector<MapPoint*> mvpLoopMPs; // 闭环候选帧的部分一二级可视KFs的所有地图点
    std::vector<MapPoint*> mvpLoopMatchedMPs; // 探测公共区域时，与LastCurrentKF共视的Map点
    bool mbMergeDetected; // true: 探测到满足Map合并要求； false: 不满足
    int mnMergeNumCoincidences;
    int mnMergeNumNotFound;
    KeyFrame* mpMergeLastCurrentKF; // 上一次发生Map合并的关键帧

    /* 满足Map合并要求时，mg2oMergeSlw是从mpCurrentKF的父帧推算得到mpCurrentKF在世界坐标系下的位姿
     此时，父帧在old Map中，当前帧属于new Map,但能推算出在old Map中当前帧的位姿，同时当前帧在new Map中有新的位姿，故利用当前帧的两个位姿，即可把old Map转换到new Map的坐标系下
    */
    g2o::Sim3 mg2oMergeSlw; // world->lastKF的位姿；mg2oMergeSlw是Map合并时父帧的位姿
    g2o::Sim3 mg2oMergeSmw;
    g2o::Sim3 mg2oMergeScw; // 当前帧在old Map(MergeMap)中的位姿, 通过父帧推算得到
    KeyFrame* mpMergeMatchedKF; // 满足Map合并时，最佳的候选关键帧； 即当前帧所在的Map与mpMergeMatchedKF所在的Map进行合并
    std::vector<MapPoint*> mvpMergeMPs;
    std::vector<MapPoint*> mvpMergeMatchedMPs;
    std::vector<KeyFrame*> mvpMergeConnectedKFs;

    g2o::Sim3 mSold_new; // 老Map合并到当前Map的Sim3旋转参数
    //-------

    long unsigned int mLastLoopKFid; // 当前帧id是mLastLoopKFid时，执行了闭环优化

    // Variables related to Global Bundle Adjustment
    bool mbRunningGBA;
    bool mbFinishedGBA;
    bool mbStopGBA;
    std::mutex mMutexGBA;
    std::thread* mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool mbFixScale;


    bool mnFullBAIdx;



    vector<double> vdPR_CurrentTime;
    vector<double> vdPR_MatchedTime;
    vector<int> vnPR_TypeRecogn;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
