# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zqy/SLAM/ORB_SLAM3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zqy/SLAM/ORB_SLAM3/evaluation

# Include any dependencies generated for this target.
include CMakeFiles/stereo_tum_vi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stereo_tum_vi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereo_tum_vi.dir/flags.make

CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.o: CMakeFiles/stereo_tum_vi.dir/flags.make
CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.o: ../Examples/Stereo/stereo_tum_vi.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zqy/SLAM/ORB_SLAM3/evaluation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.o -c /home/zqy/SLAM/ORB_SLAM3/Examples/Stereo/stereo_tum_vi.cc

CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zqy/SLAM/ORB_SLAM3/Examples/Stereo/stereo_tum_vi.cc > CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.i

CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zqy/SLAM/ORB_SLAM3/Examples/Stereo/stereo_tum_vi.cc -o CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.s

# Object files for target stereo_tum_vi
stereo_tum_vi_OBJECTS = \
"CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.o"

# External object files for target stereo_tum_vi
stereo_tum_vi_EXTERNAL_OBJECTS =

../Examples/Stereo/stereo_tum_vi: CMakeFiles/stereo_tum_vi.dir/Examples/Stereo/stereo_tum_vi.cc.o
../Examples/Stereo/stereo_tum_vi: CMakeFiles/stereo_tum_vi.dir/build.make
../Examples/Stereo/stereo_tum_vi: ../lib/libORB_SLAM3.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
../Examples/Stereo/stereo_tum_vi: /home/zqy/LIBRARY/Pangolin/build/src/libpangolin.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libGLX.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libGLU.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libGLEW.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libEGL.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libSM.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libICE.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libX11.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libXext.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libGLX.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libGLU.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libGLEW.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libEGL.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libSM.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libICE.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libX11.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libXext.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libdc1394.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libavcodec.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libavformat.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libavutil.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libswscale.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libavdevice.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/libOpenNI.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/libOpenNI2.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libpng.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libz.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libjpeg.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libtiff.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/libzstd.so
../Examples/Stereo/stereo_tum_vi: /usr/lib/x86_64-linux-gnu/liblz4.so
../Examples/Stereo/stereo_tum_vi: ../Thirdparty/DBoW2/lib/libDBoW2.so
../Examples/Stereo/stereo_tum_vi: ../Thirdparty/g2o/lib/libg2o.so
../Examples/Stereo/stereo_tum_vi: CMakeFiles/stereo_tum_vi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zqy/SLAM/ORB_SLAM3/evaluation/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../Examples/Stereo/stereo_tum_vi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereo_tum_vi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereo_tum_vi.dir/build: ../Examples/Stereo/stereo_tum_vi

.PHONY : CMakeFiles/stereo_tum_vi.dir/build

CMakeFiles/stereo_tum_vi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereo_tum_vi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereo_tum_vi.dir/clean

CMakeFiles/stereo_tum_vi.dir/depend:
	cd /home/zqy/SLAM/ORB_SLAM3/evaluation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zqy/SLAM/ORB_SLAM3 /home/zqy/SLAM/ORB_SLAM3 /home/zqy/SLAM/ORB_SLAM3/evaluation /home/zqy/SLAM/ORB_SLAM3/evaluation /home/zqy/SLAM/ORB_SLAM3/evaluation/CMakeFiles/stereo_tum_vi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereo_tum_vi.dir/depend

