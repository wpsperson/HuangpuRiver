cmake_minimum_required(VERSION 3.14)
set(ProjectName HuangpuRiver)
PROJECT(${ProjectName})

SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
SET(CMAKE_RELEASE_POSTFIX "" CACHE STRING "add a postfix, usually empty on windows")
set_property( GLOBAL PROPERTY USE_FOLDERS ON)

#ADD_SUBDIRECTORY(src)

option(BuildDemos "是否构建 Demos?" ON)
if (BuildDemos)
	ADD_SUBDIRECTORY(PlainDemos)
endif(BuildDemos)



