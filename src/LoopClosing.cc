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


#include "LoopClosing.h"

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM3
{

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0),
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

/*
 * @brief LoopClosing线程的运行函数，功能是：闭环检测全局位姿优化和Map合并-滑动窗口的BA优化
 * 过程：回环或Map合并探测，如果满足闭环要求则执行闭环优化；如果满足Map合并要求则执行Map合并
 * Map合并的原理：利用父帧到当前帧位姿的变换，父帧在old Map中，当前帧在new Map中，父帧和当前帧有共视Map点，
 * 所以通过计算父帧到当前帧的Sim3变换即可得到old Map到new  Map的变换参数
 * */
void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        //NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------
        if(CheckNewKeyFrames())
        {
            if(mpLastCurrentKF)
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear();
                mpLastCurrentKF->mvpMergeCandKFs.clear();
            }
#ifdef REGISTER_TIMES
            timeDetectBoW = 0;
            std::chrono::steady_clock::time_point time_StartDetectBoW = std::chrono::steady_clock::now();
#endif
            // 1. 探测是否满足闭环或Map合并的要求，如果满足则需要返回最佳的闭环或Map合并的候选帧
            bool bDetected = NewDetectCommonRegions(); // 在该函数中从队列头弹出的KF赋值给mpCurrentKF
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndDetectBoW = std::chrono::steady_clock::now();
            double timeDetect = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndDetectBoW - time_StartDetectBoW).count();
            double timeDetectSE3 = timeDetect - timeDetectBoW;

            if(timeDetectBoW > 0)
            {
                vTimeBoW_ms.push_back(timeDetectBoW);
            }
            vTimeSE3_ms.push_back(timeDetectSE3);
            vTimePRTotal_ms.push_back(timeDetect);
#endif

            // 2. 如果探测到满足Map合并或闭环要求
            if(bDetected)
            {
                // 2.1 如果探测到满足Map合并的要求，则执行Map合并
                if(mbMergeDetected)
                {
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO) &&
                        (!mpCurrentKF->GetMap()->isImuInitialized()))
                    {
                        // 需要等到IMU在当前Map完成初始化后才执行Map合并，原因是：IMU初始化后Map具有真实的尺度且根据重力方向已对齐到世界坐标系
                        cout << "IMU is not initilized, merge is aborted" << endl;
                    }
                    else
                    {
                        // 2.1.1 计算old Map到new Map的Sim3变换参数
                        Verbose::PrintMess("*Merged detected", Verbose::VERBOSITY_QUIET);
                        Verbose::PrintMess("Number of KFs in the current map: " + to_string(mpCurrentKF->GetMap()->KeyFramesInMap()), Verbose::VERBOSITY_DEBUG);
                        cv::Mat mTmw = mpMergeMatchedKF->GetPose();
                        g2o::Sim3 gSmw2(Converter::toMatrix3d(mTmw.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTmw.rowRange(0, 3).col(3)),1.0);
                        cv::Mat mTcw = mpCurrentKF->GetPose();
                        g2o::Sim3 gScw1(Converter::toMatrix3d(mTcw.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTcw.rowRange(0, 3).col(3)),1.0);
                        g2o::Sim3 gSw2c = mg2oMergeSlw.inverse();
                        g2o::Sim3 gSw1m = mg2oMergeSlw; // world -> last KF camera

                        mSold_new = (gSw2c * gScw1); // old Map合并到new Map的Sim3旋转参数

                        // 2.1.2 根据IMU的初始化情况，更新old Map到new Map的Sim3变换参数
                        if(mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->IsInertial())
                        {
                            if(mSold_new.scale()<0.90||mSold_new.scale()>1.1) // scale 要属于（0.9， 1.1）的区间内
                            {
                                mpMergeLastCurrentKF->SetErase();
                                mpMergeMatchedKF->SetErase();
                                mnMergeNumCoincidences = 0;
                                mvpMergeMatchedMPs.clear();
                                mvpMergeMPs.clear();
                                mnMergeNumNotFound = 0;
                                mbMergeDetected = false;
                                Verbose::PrintMess("scale bad estimated. Abort merging", Verbose::VERBOSITY_NORMAL);
                                continue;
                            }
                            // If inertial, force only yaw
                            if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO) &&
                                   mpCurrentKF->GetMap()->GetIniertialBA1()) // TODO, maybe with GetIniertialBA1
                            {
                                // IMU初始化后具有真实的尺度，并使KFs和Map与重力方向对齐，所以pitch和roll角度设为0
                                Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix()); // 对数映射， 旋转矩阵到旋转向量. phi = [roll, pitch, yaw]^T
                                phi(0)=0;
                                phi(1)=0;
                                mSold_new = g2o::Sim3(ExpSO3(phi),mSold_new.translation(),1.0); // 指数映射
                            }
                        }

                        mg2oMergeSmw = gSmw2 * gSw2c * gScw1; // Qes: 这个变换的几何意义？？？
                        mg2oMergeScw = mg2oMergeSlw;

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();
#endif
                        /*
                         * 2.1.3 执行Map合并， 分为：1）有惯性观测的Map合并；2）无惯性观测的Map合并
                         * Map合并的实质是利用父帧到当前帧位姿的变换，父帧在old Map中，当前帧在new Map中，父帧和当前帧有共视Map点，
                         * 所以通过计算父帧到当前帧的Sim3变换即可得到old Map到new Map的变换参数.
                        */
                        if (mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO)
                            MergeLocal2(); // 有惯性观测的Map合并，old Map合并到new Map中； Map合并时也会执行滑动窗口的紧耦合VI-BA位姿优化，Map合并是一种特殊的闭环优化
                        else
                            MergeLocal(); // 无惯性观测的Map合并， 当前帧所在的new Map合并到old Map
#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndMerge = std::chrono::steady_clock::now();
                        double timeMerge = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMerge - time_StartMerge).count();
                        vTimeMergeTotal_ms.push_back(timeMerge);
#endif
                    }

                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(1);

                    // Reset all variables
                    mpMergeLastCurrentKF->SetErase();
                    mpMergeMatchedKF->SetErase();
                    mnMergeNumCoincidences = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMPs.clear();
                    mnMergeNumNotFound = 0;
                    mbMergeDetected = false;

                    // 2.2 探测到同时满足Map合并和闭环，则只执行Map合并，Map合并是特殊的闭环优化
                    if(mbLoopDetected)
                    {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->SetErase();
                        mpLoopMatchedKF->SetErase();
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mvpLoopMPs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected = false;
                    }
                }

                // 2.3 探测到满足闭环要求
                if(mbLoopDetected)
                {
                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(0);

                    Verbose::PrintMess("*Loop detected", Verbose::VERBOSITY_QUIET);

                    // 2.3.1 有惯性观测的闭环矫正
                    mg2oLoopScw = mg2oLoopSlw; // 当前帧闭环计算后的目标位姿，简称为新位姿（从已有的位姿优化到mg2oLoopScw）
                    if(mpCurrentKF->GetMap()->IsInertial())
                    {
                        cv::Mat Twc = mpCurrentKF->GetPoseInverse();
                        g2o::Sim3 g2oTwc(Converter::toMatrix3d(Twc.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(Twc.rowRange(0, 3).col(3)),1.0);
                        g2o::Sim3 g2oSww_new = g2oTwc*mg2oLoopScw; // 当前帧新位姿推算到老位姿(老位姿是tracker线程跟踪得到)的差值，闭环优化就是使这个差距尽可能的小

                        Eigen::Vector3d phi = LogSO3(g2oSww_new.rotation().toRotationMatrix()); // 对数映射, 分离出欧拉角，phi = [roll, pitch, yaw]^T

                        // pitch , roll < 0.008, yaw < 0.349弧度， 航偏角yaw接近时才认为形成了闭环；yaw角差值过大会影响闭环的精度
                        if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f)
                        {
                            if(mpCurrentKF->GetMap()->IsInertial())
                            {
                                // If inertial, force only yaw
                                // 有惯性观测且当前Map已完成IMU初始化，roll和pitch置零，只优化yaw
                                if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO) &&
                                        mpCurrentKF->GetMap()->GetIniertialBA2())
                                {
                                    phi(0)=0;
                                    phi(1)=0; // IMU完全初始化后KFs和Map已经对齐到重力方向，所以roll和pitch要强制置零
                                    g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0);
                                    mg2oLoopScw = g2oTwc.inverse()*g2oSww_new;
                                }
                            }

                            mvpLoopMapPoints = mvpLoopMPs;//*mvvpLoopMapPoints[nCurrentIndex];

#ifdef REGISTER_TIMES
                            std::chrono::steady_clock::time_point time_StartLoop = std::chrono::steady_clock::now();
#endif
                            // 执行闭环矫正
                            CorrectLoop();
#ifdef REGISTER_TIMES
                            std::chrono::steady_clock::time_point time_EndLoop = std::chrono::steady_clock::now();
                            double timeLoop = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLoop - time_StartLoop).count();
                            vTimeLoopTotal_ms.push_back(timeLoop);
#endif
                        }
                        else
                        {
                            cout << "BAD LOOP!!!" << endl;
                        }
                    }
                    else
                    {
                        mvpLoopMapPoints = mvpLoopMPs;
#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartLoop = std::chrono::steady_clock::now();
#endif
                        // 2.3.2 无惯性观测下的闭环优化
                        CorrectLoop();

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndLoop = std::chrono::steady_clock::now();
                        double timeLoop = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLoop - time_StartLoop).count();
                        vTimeLoopTotal_ms.push_back(timeLoop);
#endif
                    }

                    // Reset all variables
                    mpLoopLastCurrentKF->SetErase();
                    mpLoopMatchedKF->SetErase();
                    mnLoopNumCoincidences = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMPs.clear();
                    mnLoopNumNotFound = 0;
                    mbLoopDetected = false;
                }

            } // end if(bDetected) 合并Map
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if(CheckFinish()){
            break;
        }

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

/*
 * @brief 探测是否满足闭环或Map合并的条件并找到闭环(或Map合并的)候选关键帧
 * 过程：1. 是否执行公共区域判断验证；
 * 2. 连续三个当前帧与闭环候选帧的共视地图点>100则满足闭环要求
 * 3. 连续三个当前帧与Map合并候选帧的共视地图点>100则满足Map合并的要求
 * 4. 在关键帧数据库中根据BoW搜索闭环（或Map合并）的候选KF
 * 5. 用ORB匹配和几何验证的方法探测是否满足闭环的要求，并返回最佳的候选KF
 * 6. 用ORB匹配和几何验证的方法探测是否满足Map合并的要求，并返回最佳的候选KF
 * 共视区域探测，两种：1）探测是否构成闭环； 2）探测是否需要Map合并
 * */
bool LoopClosing::NewDetectCommonRegions()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;

        mpLastMap = mpCurrentKF->GetMap();
    }

    if(mpLastMap->IsInertial() && !mpLastMap->GetIniertialBA1())
    {
        mpKeyFrameDB->add(mpCurrentKF); // 将当前帧KF插入到KeyFrameDB中
        mpCurrentKF->SetErase();
        return false; // 有惯性观测，但IMU还未做第二次初始化，说明跟踪时间较短，没必要执行闭环检测，故return
    }

    if(mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) //12
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false; // Stereo情形时，活动Map中的KF较少，也不执行闭环检测
    }

    if(mpLastMap->GetAllKeyFrames().size() < 12)
    {
        mpKeyFrameDB->add(mpCurrentKF); // Bug！ 此处连续三个if中，当前帧会被重复插入到KeyFrameDB
        mpCurrentKF->SetErase();
        return false; // 活动Map中的KF少于12帧，保存当前帧到KeyFrameDB，不执行闭环检测
    }

    // Check the last candidates with geometric validation
    // Loop candidates 探测是否构成闭环
    bool bLoopDetectedInKF = false;
    bool bCheckSpatial = false;

    // 2. 连续三个当前帧与闭环候选帧的共视地图点>100则满足闭环要求
    if(mnLoopNumCoincidences > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        cv::Mat mTcl = mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse();
        g2o::Sim3 gScl(Converter::toMatrix3d(mTcl.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTcl.rowRange(0, 3).col(3)),1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw; // 当前帧在世界坐标系下的位姿 world->currentKF
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        // mpCurrentKF与mpLoopMatchedKF共视的Map点>100则认为有公共区域bCommonRegion = true
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs);
        if(bCommonRegion)
        {
            bLoopDetectedInKF = true;
            mnLoopNumCoincidences++; // 帧间存在共视情况的次数++
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;

            mbLoopDetected = mnLoopNumCoincidences >= 3; // 连续当前关键帧mpCurrentKF(每一次都是新的KF)与mpLoopMatchedKF存在公共区域超过3次，则认为存在构成回环
            mnLoopNumNotFound = 0;
            if(!mbLoopDetected)
            {
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;
            mnLoopNumNotFound++;
            if(mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0; // 置零是为了排除偶然情况，如果存在闭环则连续的几个当前帧和候选帧间存在共视，而不是时不时有共视
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }
        }
    }

    // 3. 连续三个当前帧与Map合并候选帧的共视地图点>100则满足Map合并的要求
    bool bMergeDetectedInKF = false;
    if(mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates
        cv::Mat mTcl = mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse(); // lastKF到CurrentKF的相对位姿
        g2o::Sim3 gScl(Converter::toMatrix3d(mTcl.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTcl.rowRange(0, 3).col(3)),1.0);
        g2o::Sim3 gScw = gScl * mg2oMergeSlw; // gScw是从mpCurrentKF父帧推算得到的mpCurrentKF在世界坐标系下的位姿
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if(bCommonRegion)
        {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3; // 连续三个mpCurrentKF与mpMergeMatchedKF的共视点>100，则认为探测到Map需要合并
        }
        else
        {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if(mnMergeNumNotFound >= 2)
            {

                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0; // 置零，防止对相距较远的三次共视执行合并
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0; // 置零
            }
        }
    }

    // 如果连续3帧与候选帧存在较多共视点，则认为回到曾走过的地方
    if(mbMergeDetected || mbLoopDetected)
    {
        mpKeyFrameDB->add(mpCurrentKF); // 不满足条件时，后边仍会将mpCurrentKF加入mpKeyFrameDB(所有的mpCurrentKF均会加入mpKeyFrameDB)，这一句写在if语句之前多合理？
        return true;
    }

    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;

    // Extract candidates from the bag of words
    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    if(!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartDetectBoW = std::chrono::steady_clock::now();
#endif
        // Search in BoW
        // 4. 如果没有足够的共视点(bCommonRegion = false),在关键帧数据库中根据BoW搜索闭环（或Map合并）的候选KF
        mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCand, vpMergeBowCand,3);

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndDetectBoW = std::chrono::steady_clock::now();
        timeDetectBoW = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndDetectBoW - time_StartDetectBoW).count();
#endif
    }

    // 5. 探测闭环的候选帧，并判断是否满足闭环要求,并返回最佳的候选KF
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        // 根据共视Map点的个数判断不具备闭环条件，但根据视觉相似性判断具备闭环共视条件，则采用ORB匹配和几何验证的方法更进一步判断
        // !此处mpLoopMatchedKF会被修改为探测到的最佳的闭环候选KF； mpLoopLastCurrentKF会被修改为当前帧； mpLoopMatchedKF是最佳的闭环候选帧
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand, mpLoopMatchedKF, mpLoopLastCurrentKF, mg2oLoopSlw, mnLoopNumCoincidences, mvpLoopMPs, mvpLoopMatchedMPs);
    }

    // Merge candidates
    // 6. 探测Map合并的候选帧，并判断是否满足Map合并的要求,并返回最佳的候选KF
    if(!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        // 根据共视Map点的个数判断不具备Map合并的条件，但根据视觉相似性判断具备Map合并条件，则采用ORB匹配和几何验证的方法更进一步判断
        // ！此处mpMergeLastCurrentKF会被修改为当前帧；mpMergeMatchedKF是验证得到的最佳Map合并候选帧
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand, mpMergeMatchedKF, mpMergeLastCurrentKF, mg2oMergeSlw, mnMergeNumCoincidences, mvpMergeMPs, mvpMergeMatchedMPs);
    }

    mpKeyFrameDB->add(mpCurrentKF); // mpCurrentKF加入KeyFrameDB
    if(mbMergeDetected || mbLoopDetected)
    {
        return true;
    }

    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false; // 未探测到存在闭环或Map合并

    return false;
}

bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs;
    // 将候选匹配帧pMatchedKF的部分一二级可视KF的地图点投影到当前帧，在指定半径内匹配ORB点
    // gScw是当前帧的位姿；vpMPs是候选匹配帧pMatchedKF的部分一二级可视KF的所有地图点；vpMatchedMPs是投影匹配到的地图点
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if(nNumProjMatches >= nProjMatches)
    {
        cv::Mat mScw = Converter::toCvMat(gScw);
        cv::Mat mTwm = pMatchedKF->GetPoseInverse();
        g2o::Sim3 gSwm(Converter::toMatrix3d(mTwm.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTwm.rowRange(0, 3).col(3)),1.0);
        g2o::Sim3 gScm = gScw * gSwm; // 候选匹配pMatchedKF到当前帧的相对位姿
        Eigen::Matrix<double, 7, 7> mHessian7x7; // 后边未使用的变量

        bool bFixedScale = mbFixScale;       // TODO CHECK; Solo para el monocular inertial
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale=false; // 有惯性观测但IMU还未初始化则尺度不固定，后边优化候选匹配KF到当前KF的相对位姿时尺度scale也需要参与优化

        /* 这块的代码写的low，mpCurrentKF是指针已经从外部传入给pCurrentKF；然后这里调用OptimizeSim3()函数又用mpCurrentKF，
         * 显然破坏了DetectAndReffineSim3FromLastKF函数的独立性！
         */
        // 图优化帧间位姿的Sim3变换参数
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);
        if(numOptMatches > nProjOptMatches)
        {
            // 计算出的Sim3变换参数不用，反而还是用当前帧的位姿初值继续去投影匹配，是为了验证当前帧的位姿？而不是更新当前帧的位姿？如果验证，何必做Sim3计算？
            g2o::Sim3 gScw_estimation(Converter::toMatrix3d(mScw.rowRange(0, 3).colRange(0, 3)),
                           Converter::toVector3d(mScw.rowRange(0, 3).col(3)),1.0); // 当前帧的位姿 world->pCurrentKF camera

            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));

            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

/*
 * @brief 验证当前帧与闭环(或Map合并)候选帧间是否存在闭环或满足Map合并的共视要求
 * 过程是，采用BoW视觉匹配到的Map点再用几何验证的方法进行验证，验证当前帧中Map点被候选及候选的共视帧的可见性，
 * 这也会验证当前帧位姿的精准度。（该函数的实现，逻辑真是好不Elegant！可尝试更简洁有效的办法！）
 * 注意，当检查闭环候选KF调用该函数时可能会修改mpLoopLastCurrentKF为当前帧；检查Map合并的候选KF时，mpMergeLastCurrentKF可能会被修改为当前帧
 * 即，类中成员变量mpLoopLastCurrentKF和mpMergeLastCurrentKF值的更新是在该函数中
 * vpBowCand：根据视觉相似性找到的闭环或Map合并的候选KF；
 * pMatchedKF2：最佳候选
 * */
bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 5;

    ORBmatcher matcherBoW(0.9, true); // 距离比例ratio取0.9， 宽松
    ORBmatcher matcher(0.75, true);
    int nNumGuidedMatching = 0;

    KeyFrame* pBestMatchedKF;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);

    int index = 0;
    for(KeyFrame* pKFi : vpBowCand)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        // Current KF against KF with covisibles version
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles); // 当前帧的闭环（或Map合并）候选帧的5个最佳共视帧
        vpCovKFi.push_back(vpCovKFi[0]); // 把pHKi的最佳共视的第一个KF放到最后，并把第一个换成pHKi
        vpCovKFi[0] = pKFi;

        std::vector<std::vector<MapPoint*> > vvpMatchedMPs; // 当前帧与pKFi(pKFi是当前帧的闭环或Map合并的候选KF)的nNumCovisibles个最佳共视帧的公共Map点（通过Bow加速下的ORB匹配得到）
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi; // 用于标记找到的公共Map点是否重复保存
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0; // 当前帧与vpCovKFi中的元素具有最多的Map点数

        // vpMatchedPoints是当前帧与视觉相似帧具有的共同Map点
        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
        // vpKeyFrameMatchedMP是与当前帧不仅视觉相似且具有公共的Map点
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));

        int nIndexMostBoWMatchesKF=0; // 具有最多公共Map点的KFi帧在vpCovKFi数组中的id

        /* 1. 寻找当前帧与vpCovKFi中KFi共视的Map点及共视KF，保存至vpMatchedPoints和vpKeyFrameMatchedMP */
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                continue;

            // 当前帧和vpCovKFi[j]在BoW加速下ORB匹配，得到公共的Map点的数组vvpMatchedMPs[j]
            int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]); // vvpMatchedMPs[j]的size是mpCurrentKF的Map点数
            if (num > nMostBoWNumMatches)
            {
                nMostBoWNumMatches = num;
                nIndexMostBoWMatchesKF = j;
            }
        }

        bool bAbortByNearKF = false;
        for(int j=0; j<vpCovKFi.size(); ++j)
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                // vpCovKFi[j]不是当前帧的共视帧，则不考虑vpCovKFi[j]位候选KF
                bAbortByNearKF = true;
                break;
            }

            for(int k=0; k < vvpMatchedMPs[j].size(); ++k)
            {
                MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                if(!pMPi_j || pMPi_j->isBad())
                    continue;

                if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                {
                    spMatchedMPi.insert(pMPi_j);
                    numBoWMatches++; // 找到公共Map点的点数

                    vpMatchedPoints[k]= pMPi_j; // 加入到所有匹配的Map点的数组中
                    vpKeyFrameMatchedMP[k] = vpCovKFi[j]; // 有根据BoW判断具有视觉相似性，且与当前帧具有公共的Map点
                }
            }
        }

        /* 2. 几何验证当前帧的可靠性，包括共视性和位姿的精度 */
        if(!bAbortByNearKF && numBoWMatches >= nBoWMatches) // 邻近共视帧可用且找到足够数量的Map点          // TODO pick a good threshold
        {
            // Geometric validation
            /* 2.1 计算mpCurrentKF与pKi间位姿的Sim3参数，并验证 */
            bool bFixedScale = mbFixScale;
            if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                bFixedScale=false;

            // 计算pMostBoWMatchesKF到mpCurrentKF的Sim3相对位姿gScm
            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers， 最多迭代300次

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers;
            bool bConverge = false;
            cv::Mat mTcm; // mpCurrentKF与pKi间位姿的Sim3参数
            while(!bConverge && !bNoMore)
            {
                mTcm = solver.iterate(20,bNoMore, vbInliers, nInliers, bConverge);
            }

            if(bConverge)
            {
                // 如果收敛，则重新收集可能共视的地图点，用于下一次验证
                vpCovKFi.clear();
                vpCovKFi = pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                int nInitialCov = vpCovKFi.size();
                vpCovKFi.push_back(pMostBoWMatchesKF);
                set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                set<MapPoint*> spMapPoints;
                vector<MapPoint*> vpMapPoints;
                vector<KeyFrame*> vpKeyFrames;
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij); // 邻近共视帧的所有Map点
                            vpKeyFrames.push_back(pCovKFi); // 所有的邻近共视帧
                        }
                    }
                }

                /* 2.2 再一次验证 */
                g2o::Sim3 gScm(Converter::toMatrix3d(solver.GetEstimatedRotation()),Converter::toVector3d(solver.GetEstimatedTranslation()),solver.GetEstimatedScale());
                g2o::Sim3 gSmw(Converter::toMatrix3d(pMostBoWMatchesKF->GetRotation()),Converter::toVector3d(pMostBoWMatchesKF->GetTranslation()),1.0);
                g2o::Sim3 gScw = gScm*gSmw; // 根据候选帧的位姿gSmw和候选-当前帧的相对位姿gScm推算得到当前帧的矫正位姿(闭环时，是当前帧的目标位姿)
                cv::Mat mScw = Converter::toCvMat(gScw);

                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);

                if(numProjMatches >= nProjMatches)
                {
                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale = mbFixScale;
                    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                        bFixedScale=false;

                    // 进一步优化vpMatchedMP到mpCurrentKF的Sim3相对位姿参数
                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, mHessian7x7, true);

                    if(numOptMatches >= nSim3Inliers)
                    {
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pMostBoWMatchesKF->GetRotation()),Converter::toVector3d(pMostBoWMatchesKF->GetTranslation()),1.0);
                        g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position 当前帧在世界坐标系下的位姿
                        cv::Mat mScw = Converter::toCvMat(gScw); // 当前帧的位姿

                        vector<MapPoint*> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);

                        if(numProjOptMatches >= nProjOptMatches) // >= 80
                        {
                            // 当前帧与候选的KF共视的地图点超过80个则认为稳定（准确来说是，当前帧中超过80个地图点都可被候选及候选的共视帧可见）
                            int nNumKFs = 0;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                            int j = 0;
                            while(nNumKFs < 3 && j<vpCurrentCovKFs.size())
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                cv::Mat mTjc = pKFj->GetPose() * mpCurrentKF->GetPoseInverse(); // 当前帧到pKFj的相对位姿
                                g2o::Sim3 gSjc(Converter::toMatrix3d(mTjc.rowRange(0, 3).colRange(0, 3)),Converter::toVector3d(mTjc.rowRange(0, 3).col(3)),1.0);
                                g2o::Sim3 gSjw = gSjc * gScw; // pKFj的位姿
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                // pKFj与pMostBoWMatchesKF共视超过30个地图点，则bValid = true
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj,pMostBoWMatchesKF, gSjw,numProjMatches_j, vpMapPoints, vpMatchedMPs_j);
                                if(bValid)
                                {
                                    nNumKFs++;
                                }
                                j++;
                            }

                            if(nNumKFs < 3)
                            {
                                vnStage[index] = 8;
                                vnMatchesStage[index] = nNumKFs;
                            }

                            if(nBestMatchesReproj < numProjOptMatches)
                            {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                g2oBestScw = gScw;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }

                    } // end - if (numOptMatches >= nSim3Inliers)

                }  // end -if(numProjMatches >= nProjMatches)
            }
        }
        index++;
    } // end for(KeyFrame* pKFi : vpBowCand)

    if(nBestMatchesReproj > 0)
    {
        // 如果有足够的共视点且当前帧的可视性（被其他帧同时的可见性）较好，则pLastCurrentKF替换为当前帧
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        return nNumCoincidences >= 3;
    }
    else
    {
        int maxStage = -1;
        int maxMatched;
        for(int i=0; i<vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches)
    {
        return true;
    }
    return false;
}

/*
 * @brief 将候选匹配帧pMatchedKFw的部分一二级可视KF的地图点投影到当前帧，在指定半径内匹配ORB点
 * */
int LoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                         set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                         vector<MapPoint*> &vpMatchedMapPoints)
{
    int nNumCovisibles = 5;
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles); // 候选匹配KF的数组，pMatchedKFw的一二级的部分可视帧帧
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();
    // 将pMatchedKFw的部分一二级邻近（不属于一级邻近）加入vpCovKFm数组
    for(int i=0; i<nInitialCov; ++i)
    {
        vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles); // MatchedKF的二级扩展
        int nInserted = 0;
        int j = 0;
        while(j < vpKFs.size() && nInserted < nNumCovisibles)
        {
            if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
            {
                spCheckKFs.insert(vpKFs[j]); // 新的二级MatchedKF
                ++nInserted;
            }
            ++j;
        }
        vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
    }
    set<MapPoint*> spMapPoints; // 用于保存vpCovKFm中所有的KF的地图点， 这个变量后边没有用
    vpMapPoints.clear(); // 保存vpCovKFm中所有的KF的地图点
    vpMatchedMapPoints.clear();
    for(KeyFrame* pKFi : vpCovKFm)
    {
        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    cv::Mat mScw = Converter::toCvMat(g2oScw); // 当前帧pCurrentKF的位姿

    ORBmatcher matcher(0.9, true); // 最小和次最小的距离比例阈值ratio值取0.9

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL)); // size is 当前帧的Map点数，用赋NULL的办法避免数组中元素的删除处理，高效
    // 将两级候选匹配关键帧的地图点投影到当前帧，在指定半径内匹配ORB点
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

/*
 * @brief 闭环矫正的执行函数
 * */
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue(); // Proccess keyframes in the queue

    // If a Global Bundle Adjustment is running, abort it
    // 1. 暂停GBA和LocalMapping
    cout << "Request GBA abort" << endl;
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            cout << "GBA running... Abort!" << endl;
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    cout << "start updating connections" << endl;
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    // 2. 遍历与当前帧相连的KF, 根据当前帧矫正后的位姿， 和当前帧与连接帧间的相对位姿计算连接帧矫正后的位姿
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oLoopScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();
    Map* pLoopMap = mpCurrentKF->GetMap();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);
        const bool bImuInit = pLoopMap->isImuInitialized();
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            cv::Mat Tiw = pKFi->GetPose();
            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc; // mpCurrentKF -> pKFi
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oLoopScw; // pKFi矫正后的位姿
                // Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            // Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        // 3. 计算被当前帧及其邻近帧可视的MPs的矫正后的坐标
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw)); // 计算矫正后的MPs的坐标； world->camera; camera->new world

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]
            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);
            pKFi->SetPose(correctedTiw); // KFi的位姿更新为矫正后的位姿

            // Correct velocity according to orientation correction
            if(bImuInit)
            {
                Eigen::Matrix3d Rcor = eigR.transpose()*g2oSiw.rotation().toRotationMatrix();
                pKFi->SetVelocity(Converter::toCvMat(Rcor)*pKFi->GetVelocity());
            }

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }
        // TODO Check this index increasement
        pLoopMap->IncreaseChangeIndex();

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        // 4. 参与闭环优化的MPs的融合， 旧坐标换为新坐标
        for(size_t i=0; i<mvpLoopMatchedMPs.size(); i++)
        {
            if(mvpLoopMatchedMPs[i])
            {
                MapPoint* pLoopMP = mvpLoopMatchedMPs[i]; // mvpLoopMatchedMPs是以当前帧的MPs为查询点匹配得到
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // 5. Fuse duplications.
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    // 6. MPs融合后，出现新的连接，从新的帧间连接中中删除旧的连接  Qes: 为何要这么做？？？
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections(); // 更新pKFi的公式关系，得到新的共视连接的KFs
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        // 从新的共视KFs中删除旧的共视KFs
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // 7. Optimize graph
    bool bFixedScale = mbFixScale;
    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
        bFixedScale=false; // 只有IMU初始化彻底完成时，尺度才不参与优化

    // 基于本质图的位姿图优化
    if(pLoopMap->IsInertial() && pLoopMap->isImuInitialized())
    {
        // 7.1 IMU模式下是4-DoF优化， scale, roll, pitch不参与优化
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        // 7.2 普通模式下是7-DoF优化
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }

    mpAtlas->InformNewBigChange();

    // Add loop edge
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    if(!pLoopMap->isImuInitialized() || (pLoopMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1))
    {
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        // 8. IMU还未初始化，或Map中关键帧的数量小于200帧，则执行Global BA
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->mnId);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();
    mLastLoopKFid = mpCurrentKF->mnId; //TODO old varible, it is not use in the new algorithm
}

/*
 * @brief 无惯性观测的Map合并，是将当前帧所在的Map(new Map)合并到候选帧所在的Map中(old Map)， 即new Map -> old Map
 * */
void LoopClosing::MergeLocal()
{
    Verbose::PrintMess("MERGE: Merge Visual detected!!!!", Verbose::VERBOSITY_NORMAL);

    int numTemporalKFs = 15;

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    Verbose::PrintMess("MERGE: Check Full Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
    // If a Global Bundle Adjustment is running, abort it
    // 1. Map合并前先停止Global-BA
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }

    Verbose::PrintMess("MERGE: Request Stop Local Mapping", Verbose::VERBOSITY_DEBUG);
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    // 2. Map合并前先停止LocalMapping线程
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    Verbose::PrintMess("MERGE: Local Map stopped", Verbose::VERBOSITY_DEBUG);

    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // 3. 添加参加滑动窗口BA优化的KF和地图点
    //Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<KeyFrame*> spLocalWindowKFs; // 在无惯性观测情况下，存储的是mpCurrentKF及其部分最佳共视帧, 即spLocalWindowKFs中的关键帧均在new Map中
    //Get MPs in the welding area from the current map
    set<MapPoint*> spLocalWindowMPs; // spLocalWindowMPs中的Map点是在new Map中
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        // 有惯性观测，将mpCurrentKF的上溯numTemporalKFs个父帧及其地图点加入局部窗口的KF数组
        KeyFrame* pKFi = mpCurrentKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            // pKFi = mpCurrentKF->mPrevKF; // 原有代码，逻辑错误
            pKFi = pKFi->mPrevKF; // zqy Add 20210712
            nInserted++;

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->mNextKF;
        while(pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
            pKFi = pKFi->mNextKF; // 避免死循环, zqy Add 20210713
        }
    }
    else
    {
        // 无惯性观测，仅将当前帧加入局部窗口的KF数组
        spLocalWindowKFs.insert(mpCurrentKF);
    }

    vector<KeyFrame*> vpCovisibleKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end()); // 将当前KF的最佳共视的KFs加入spLocalWindowKFs
    const int nMaxTries = 3;
    int nNumTries = 0;
    // 将spLocalWindowKFs中KFi的部分最佳共视的KF加入到spLocalWindowKFs(局部关键帧数组)
    while(spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        vpNewCovKFs.empty();
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }
            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    // 关键帧KFi加入局部关键帧数组了，KFi的Map点也要加入，后边参与Sliding Window BA优化
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        set<MapPoint*> spMPs = pKFi->GetMapPoints();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    set<KeyFrame*> spMergeConnectedKFs; // 在无惯性观测情况下，存储的是mpMergeMatchedKF及其部分最佳共视帧
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial())
    {
        KeyFrame* pKFi = mpMergeMatchedKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            // pKFi = mpCurrentKF->mPrevKF; // 原有代码 逻辑错误
             pKFi = pKFi->mPrevKF; // 原有代码 逻辑错误
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->mNextKF;
        while(pKFi)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = pKFi->mNextKF; // 避免死循环， zqy Add 20210713
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF); // 无关性观测仅将mpMergeMatchedKF加入
    }

    vpCovisibleKFs = mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    nNumTries = 0;
    while(spMergeConnectedKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        for(KeyFrame* pKFi : spMergeConnectedKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }
            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<MapPoint*> spMapPointMerge; // 与候选KF相连的KFs上的MapPoints
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
    }

    vector<MapPoint*> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // 4. Map合并
    // 4.1 计算选定KF在合并后Map中的位姿
    // 计算当前帧的
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
    cv::Mat twc = Twc.rowRange(0,3).col(3);
    g2o::Sim3 g2oNonCorrectedSwc(Converter::toMatrix3d(Rwc),Converter::toVector3d(twc),1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = mg2oMergeScw;

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF]=g2oCorrectedScw; // mpCurrentKF在Map合并后的位姿
    vNonCorrectedSim3[mpCurrentKF]=g2oNonCorrectedScw; // mpCurrentKF在Map合并前的位姿， 也即是将new Map合并到pMergeMap中
    // 计算其他连接帧pKFi合并后的位姿
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        g2o::Sim3 g2oCorrectedSiw;
        if(pKFi!=mpCurrentKF)
        {
            cv::Mat Tiw = pKFi->GetPose();
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            vNonCorrectedSim3[pKFi]=g2oSiw;

            cv::Mat Tic = Tiw*Twc; // mpCurrentKF->pKFi的相对位姿
            cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
            cv::Mat tic = Tic.rowRange(0,3).col(3);
            g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
            g2oCorrectedSiw = g2oSic*mg2oMergeScw; // 推算得到pKFi帧在MergeMap中的位姿
            vCorrectedSim3[pKFi]=g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge = pKFi->GetPose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
        double s = g2oCorrectedSiw.scale();

        pKFi->mfScale = s;
        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

        pKFi->mTcwMerge = correctedTiw; // 合并后，在合并后Map上的位姿

        if(pCurrentMap->isImuInitialized())
        {
            Eigen::Matrix3d Rcor = eigR.transpose()*vNonCorrectedSim3[pKFi].rotation().toRotationMatrix(); // 合并后到合并前的相对位姿
            pKFi->mVwbMerge = Converter::toCvMat(Rcor)*pKFi->GetVelocity();
        }
    }

    // 4.2 计算选定MPs在合并后Map中的坐标
    for(MapPoint* pMPi : spLocalWindowMPs)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        cv::Mat P3Dw = pMPi->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(eigP3Dw)); // old world->camera；camera->new world; .map() : return s*(r*xyz) + t;
        Eigen::Matrix3d eigR = g2oCorrectedSwi.rotation().toRotationMatrix();
        Eigen::Matrix3d Rcor = eigR * g2oNonCorrectedSiw.rotation().toRotationMatrix();

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);

        pMPi->mPosMerge = cvCorrectedP3Dw; // Map合并后该点的坐标
        pMPi->mNormalVectorMerge = Converter::toCvMat(Rcor) * pMPi->GetNormal(); // 没有更新计算该点的视角向量，为何还要赋值？？？
    }

    {
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            if(!pKFi || pKFi->isBad())
            {
                continue;
            }

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTcwMerge); // 更新KFi的位姿

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap); // pKFio是在new Map中,需要转到pMergeMap, 断开与pCurrentMap的联系
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);
            if(pCurrentMap->isImuInitialized())
            {
                pKFi->SetVelocity(pKFi->mVwbMerge);
            }
        }

        // 5. 更新被选中MPs的坐标，并与原Map断开联系，合并到pMergeMap中
        for(MapPoint* pMPi : spLocalWindowMPs)
        {
            if(!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap); // 使得pCurrentMap不可用
        pMergeMap->IncreaseChangeIndex();
    }

    // Rebuild the essential graph in the local window
    pCurrentMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpCurrentKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF); // mpMergeMatchedKF设为mpCurrentKF的父帧
    // 6. 将mpCurrentKF的父子帧关系颠倒，即以前的父帧变为mpCurrentKF的子帧，父父帧变为mpCurrentKF的子子帧，....
    while(pNewChild )
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();
        pNewChild->ChangeParent(pNewParent);
        pNewParent = pNewChild;
        pNewChild = pOldParent;
    }

    //Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();

    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    // 7. Map点融合
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);

    // Update connectivity
    // 地图点变动了，需要更新KF间的连接关系
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;
        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;
        pKFi->UpdateConnections();
    }

    bool bStop = false;
    Verbose::PrintMess("MERGE: Start local BA ", Verbose::VERBOSITY_DEBUG);
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    // 8. 执行滑动窗口的局部BA优化
    if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO)
    {
        // 有惯性观测时是紧耦合滑动窗口VI-BA优化
        Optimizer::MergeInertialBA(mpLocalMapper->GetCurrKF(),mpMergeMatchedKF,&bStop, mpCurrentKF->GetMap(),vCorrectedSim3);
    }
    else
    {
        // 仅视觉观测下是局部视觉BA优化
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs,&bStop); // vpMergeConnectedKFs是fixed KFs
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    Verbose::PrintMess("MERGE: Finish the LBA", Verbose::VERBOSITY_DEBUG);

    //Update the non critical area from the current map to the merged map
    /*
     * 9. 如果在pCurrentMap中还有KF没被选中，即还未被合并到pMergeMap地图中，则需要将这些KF的位姿及Map点的坐标进行变换与pMergeMap对齐，
     * 然后执行基于本质图的位姿优化，最后将这些KF和MPs全部合并到pMergeMap地图中。即超出合并区的处理不做BA只执行基于本质图的位姿优化
    */
    vector<KeyFrame*> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPoint*> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();

    if(vpCurrentMapKFs.size() == 0)
    {
        Verbose::PrintMess("MERGE: There are not KFs outside of the welding area", Verbose::VERBOSITY_DEBUG);
    }
    else
    {
        Verbose::PrintMess("MERGE: Calculate the new position of the elements outside of the window", Verbose::VERBOSITY_DEBUG);
        //Apply the transformation
        {
            if(mpTracker->mSensor == System::MONOCULAR)
            {
                unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information

                // 9.1 更新KFi的位姿到pMergeMap
                for(KeyFrame* pKFi : vpCurrentMapKFs)
                {
                    if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                    {
                        continue;
                    }

                    g2o::Sim3 g2oCorrectedSiw;

                    cv::Mat Tiw = pKFi->GetPose();
                    cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
                    cv::Mat tiw = Tiw.rowRange(0,3).col(3);
                    g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
                    //Pose without correction
                    vNonCorrectedSim3[pKFi]=g2oSiw;

                    cv::Mat Tic = Tiw*Twc;
                    cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                    cv::Mat tic = Tic.rowRange(0,3).col(3);
                    g2o::Sim3 g2oSim(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                    g2oCorrectedSiw = g2oSim*mg2oMergeScw;
                    vCorrectedSim3[pKFi]=g2oCorrectedSiw;

                    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                    Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                    Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                    double s = g2oCorrectedSiw.scale();

                    pKFi->mfScale = s;
                    eigt *=(1./s); //[R t/s;0 1]

                    cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

                    pKFi->mTcwBefMerge = pKFi->GetPose();
                    pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
                    pKFi->SetPose(correctedTiw);

                    if(pCurrentMap->isImuInitialized())
                    {
                        Eigen::Matrix3d Rcor = eigR.transpose()*vNonCorrectedSim3[pKFi].rotation().toRotationMatrix();
                        pKFi->SetVelocity(Converter::toCvMat(Rcor)*pKFi->GetVelocity()); // TODO: should add here scale s
                    }

                }
                // 9.2 更新MPs到pMergeMap
                for(MapPoint* pMPi : vpCurrentMapMPs)
                {
                    if(!pMPi || pMPi->isBad()|| pMPi->GetMap() != pCurrentMap)
                        continue;

                    KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
                    g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                    g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                    Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);
                    pMPi->UpdateNormalAndDepth();
                }
            }
        }

        mpLocalMapper->RequestStop();
        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        // 基于本质图的位姿优化
        if(mpTracker->mSensor != System::MONOCULAR)
        {
            // vpMergeConnectedKFs是FixedKFs, vpLocalCurrentWindowKFs是FixedCorrectedKFs, vpCurrentMapKFs是NonFixedKFs, vpCurrentMapMPs是NonCorrectedMPs
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }

        // 将pCurrentMap地图中剩余的KF和MPs添加到pMergeMap地图中
        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }

                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }

            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }
        }
    } // end-else 更新超出pCurrentMap合并区的KFs和MPs的位姿和坐标

    mpLocalMapper->Release();
    Verbose::PrintMess("MERGE:Completed!!!!!", Verbose::VERBOSITY_DEBUG);

    // 10. 启动新的线程在pMergeMap中执行Global BA
    // Qes: pMergeMap执行GBA为何要判断pCurrentMap的状态？？
    if(bRelaunchBA && (!pCurrentMap->isImuInitialized() || (pCurrentMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        Verbose::PrintMess("Relaunch Global BA", Verbose::VERBOSITY_DEBUG);
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this, pMergeMap, mpCurrentKF->mnId);
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();
}

/*
 * @brief 有惯性观测下的Map合并
 * */
void LoopClosing::MergeLocal2()
{
    cout << "Merge detected!!!!" << endl;

    int numTemporalKFs = 11; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    cout << "Check Full Bundle Adjustment" << endl;
    // If a Global Bundle Adjustment is running, abort it
    // 1. Map合并前需要先停止BA的执行
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }

    // 2. Map合并需要先停止LocalMapper线程
    cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }
    cout << "Local Map stopped" << endl;

    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    // 3. old Map -> new  Map的变换
    {
        // old Map 到new Map变换的尺度、旋转及平移参数
        float s_on = mSold_new.scale();
        cv::Mat R_on = Converter::toCvMat(mSold_new.rotation().toRotationMatrix());
        cv::Mat t_on = Converter::toCvMat(mSold_new.translation());

        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

        // 3.1 将LocalMapper线程中新的KF处理完, 然后mpLocalMapper线程处于等待状态
        mpLocalMapper->EmptyQueue();

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        bool bScaleVel=false;
        if(s_on!=1)
            bScaleVel=true;

        // 3.2 old Map变换： 将mpAtlas中的当前地图变换到当前帧所在的Map下（Map合并）， new Map此时还未被保存到mpAtlas中
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(R_on,s_on,bScaleVel,t_on);
        // 3.3 更新tracking线程中上一帧和当前帧的位姿，及其对应时刻IMU的P,R,V（Map合并后，IMU要在new Map中继续进行推算位姿，故IMU的位姿也需要改变）
        mpTracker->UpdateFrameIMU(s_on,mpCurrentKF->GetImuBias(),mpTracker->GetLastKeyFrame());

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }

    const int numKFnew=pCurrentMap->KeyFramesInMap();

    // 4. 如果当前Map中IMU初始化未完全完成， 则继续IMU初始化，偏置初值设为0后，采用仅惯性的优化
    if((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO)&& !pCurrentMap->GetIniertialBA2())
    {
        // Map is not completly initialized  IMU第二次初始化未执行，IMU还未完全初始化
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        // 执行Only-Inertial的优化, 偏置被更新后需要执行优化
        Optimizer::InertialOptimization(pCurrentMap,bg,ba);
        IMU::Bias b (ba[0],ba[1],ba[2],bg[0],bg[1],bg[2]);
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        mpTracker->UpdateFrameIMU(1.0f,b,mpTracker->GetLastKeyFrame()); // IMU还未完全初始化，则传入的偏置的值均为0（IMU初始化时会计算偏置项）

        // Set map initialized
        pCurrentMap->SetIniertialBA2();
        pCurrentMap->SetIniertialBA1();
        pCurrentMap->SetImuInitialized(); // IMU的P, V， scale, gravity, bg, ba均优化，则完成IMU初始化
    }

    // Load KFs and MPs from merge map
    // 5. 将old Map中的KFs和Maps加入到当前Map中
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        vector<KeyFrame*> vpMergeMapKFs = pMergeMap->GetAllKeyFrames();
        vector<MapPoint*> vpMergeMapMPs = pMergeMap->GetAllMapPoints();

        // old Map中KF添加到new Map
        for(KeyFrame* pKFi : vpMergeMapKFs)
        {
            if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergeMap)
            {
                continue;
            }
            // Make sure connections are updated
            pKFi->UpdateMap(pCurrentMap);
            pCurrentMap->AddKeyFrame(pKFi);
            pMergeMap->EraseKeyFrame(pKFi);
        }

        // old Map中的地图点添加到new Map
        for(MapPoint* pMPi : vpMergeMapMPs)
        {
            if(!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergeMap)
                continue;

            pMPi->UpdateMap(pCurrentMap);
            pCurrentMap->AddMapPoint(pMPi);
            pMergeMap->EraseMapPoint(pMPi);
        }

        // Save non corrected poses (already merged maps)
        // 保存变换后的位姿
        vector<KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames();
        for(KeyFrame* pKFi : vpKFs)
        {
            cv::Mat Tiw=pKFi->GetPose();
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            NonCorrectedSim3[pKFi]=g2oSiw; // NonCorrectedSim3数组后边未被使用
        }
    }

    // 6. 改变Map合并候选帧的父帧间的父子关系，关系颠倒，即父帧变为子帧
    pMergeMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpMergeMatchedKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->ChangeParent(mpCurrentKF); // mpMergeMatchedKF(满足Map合并要求时最佳的候选帧，在old Map)变为mpCurrentKF的子帧
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();
        pNewChild->ChangeParent(pNewParent);
        pNewParent = pNewChild;
        pNewChild = pOldParent;
    }

    vector<MapPoint*> vpCheckFuseMapPoint; // MapPoint vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<KeyFrame*> vpCurrentConnectedKFs; // 与当前KF存在连接的KF的数组

    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vector<KeyFrame*> aux = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end());
    if (mvpMergeConnectedKFs.size()>6)
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin()+6,mvpMergeConnectedKFs.end()); // 只保留6个和mpMergeMatchedKF相连的KF(包括mpMergeMatchedKF自己)

    mpCurrentKF->UpdateConnections();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    aux = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size()>6)
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin()+6,vpCurrentConnectedKFs.end());

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
        if(spMapPointMerge.size()>1000)
            break;
    }

    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // 7. Map点融合
    // 将vpCheckFuseMapPoint中的地图点投影到vpCurrentConnectedKFs中的pKFi，投影-匹配到同名点后，vpMapPoints中的Mapi被对应于pKFi中的Mapj代替
    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint); // vpCheckFuseMapPoint中的地图点是mpMergeMatchedKF及其相连的KF可视的Map点

    // 8. 与KF关联的Map点改变后，需要更新KF的连接信息
    for(KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    if (numKFnew<10)
    {
        mpLocalMapper->Release();
        return;
    }

    // Perform BA
    // 9. 执行滑动窗口的紧耦合VI-BA, 有IMU观测的地图合并，所以执行滑动窗口的局部VI-BA, 优化合并后的Map
    bool bStopFlag=false;
    KeyFrame* pCurrKF = mpTracker->GetLastKeyFrame();
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap,CorrectedSim3);

    // Release Local Mapping.
    /* 在Map合并的过程中mpLocalMapper线程处于等待状态，合并结束后清空mpLocalMapper线程中的mlNewKeyFrames数组，
     * 实际mlNewKeyFrames数组已经被清空，合并前执行了mpLocalMapper->EmptyQueue()函数
     */
    mpLocalMapper->Release();
    return;
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKFi = mit->first;
        Map* pMap = pKFi->GetMap();

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        int numFused = matcher.Fuse(pKFi,cvScw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {


                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);

            }
        }

        total_replaces += num_replaces;
    }
}


void LoopClosing::SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    for(auto mit=vConectedKFs.begin(), mend=vConectedKFs.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKF = (*mit);
        Map* pMap = pKF->GetMap();
        cv::Mat cvScw = pKF->GetPose();

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,vpMapPoints,4,vpReplacePoints); // 将vpMapPoints中的地图点投影到pKF，投影-匹配到同名点后，vpMapPoints中的Mapi被对应于pKF中的Mapj代替

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);
            }
        }
        total_replaces += num_replaces;
    }
}



void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetActiveMapRequested)
                break;
        }
        usleep(3000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
    }
    else if(mbResetActiveMapRequested)
    {

        for (list<KeyFrame*>::const_iterator it=mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            KeyFrame* pKFi = *it;
            if(pKFi->GetMap() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
                ++it;
        }

        mLastLoopKFid=mpAtlas->GetLastInitKFid();
        mbResetActiveMapRequested=false;

    }
}

/*
 * @brief 执行Global BA的函数
 * 两种情况：1）IMU还未初始化则执行Only-Vision的Global BA； 2）IMU已经初始化则执行紧耦合的Global VI-BA
 * @param nLoopKF  是调用该函数时，当前帧的id， 也即在id为nLoopKF的当前帧时做了闭环矫正
 * */
void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);

        const bool bImuInit = pActiveMap->isImuInitialized();

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartFGBA = std::chrono::steady_clock::now();
#endif

        // 1. 执行GBA, 根据有无IMU观测分为两种模式的GBA
        if (!bImuInit)
            // IMU还未初始化则执行Only-Vision的Global BA
            Optimizer::GlobalBundleAdjustemnt(pActiveMap, 10, &mbStopGBA, nLoopKF, false);
        else {
            // IMU已经初始化则执行紧耦合的Global VI-BA
            Optimizer::FullInertialBA(pActiveMap, 7, false, nLoopKF, &mbStopGBA);
        }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartMapUpdate = std::chrono::steady_clock::now();

    double timeFullGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartMapUpdate - time_StartFGBA).count();
    vTimeFullGBA_ms.push_back(timeFullGBA);
#endif


    int idx =  mnFullBAIdx;

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    /*
     * 注意，为了保证系统的实时性，在执行Global BA时，tracking和LocalMapping线程不会停止，所以GBA执行时仍有新的KF和MP从LocalMapping线程
     * 传送至LoopClosing线程，新传入的KF和MP的位姿的更新是根据其父帧和参考关键帧的位姿
    */
    // 2. 地图中所有KFs的位姿，IMU的速度，偏置及MPs坐标的更新
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!bImuInit && pActiveMap->isImuInitialized())
            return;

        if(!mbStopGBA)
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(),pActiveMap->mvpKeyFrameOrigins.end());
            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad())
                        continue;

                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        // 矫正未参与Global BA优化的KF的位姿
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA; // LocalMapper线程新传入KF的被矫正后的位姿

                        cv::Mat Rcor = pChild->mTcwGBA.rowRange(0,3).colRange(0,3).t()*pChild->GetRotation();
                        if(!pChild->GetVelocity().empty()){
                            pChild->mVwbGBA = Rcor*pChild->GetVelocity();
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);

                        pChild->mBiasGBA = pChild->GetImuBias();
                        pChild->mnBAGlobalForKF=nLoopKF;
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA); // 更新位姿，到这里，所有地图中所有KFs的位姿才被更新为BA优化后的结果！！
                if(pKF->bImu)
                {
                    pKF->mVwbBefGBA = pKF->GetVelocity();
                    if (pKF->mVwbGBA.empty())
                        Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);                    
                }
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();
            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];
                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    // 未参与BA优化的MPs坐标的更新是根据参考KF
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    if(pRefKF->mTcwBefGBA.empty())
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);
                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            mpLocalMapper->Release();
            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndMapUpdate = std::chrono::steady_clock::now();

    double timeMapUpdate = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMapUpdate - time_StartMapUpdate).count();
    vTimeMapUpdate_ms.push_back(timeMapUpdate);

    double timeGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMapUpdate - time_StartFGBA).count();
    vTimeGBATotal_ms.push_back(timeGBA);
#endif
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
