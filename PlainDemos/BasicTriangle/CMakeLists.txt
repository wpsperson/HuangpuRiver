set(TARGET_NAME BasicTriangle)

INCLUDE_DIRECTORIES(${DemoPath})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/glm-0.9.9.8)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/glfw-3.2.1/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/stb/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/tinyobjloader/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/vulkan-1.3/include)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/glfw-3.2.1/lib)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/ThirdParty/vulkan-1.3/lib)

set(Target_HEADER_SRC_LIST
${DemoPath}/${TARGET_NAME}/main.cpp
)

#FILE(GLOB Target_SHADER_LIST ${PlainExamPath}/${TARGET_NAME}/shaders/*.glsl)
#source_group("shader" FILES ${Target_SHADER_LIST})
# ${Target_SHADER_LIST}
add_executable(${TARGET_NAME}  ${Target_HEADER_SRC_LIST} )

set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")	
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
set_target_properties(${TARGET_NAME} PROPERTIES RELEASE_POSTFIX "${CMAKE_RELEASE_POSTFIX}")
#set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "PlainOpenGLExamples")

SET(ADDITIONAL_LIBRARIES_DEBUG
vulkan-1.lib
glfw3d.lib
)
SET(ADDITIONAL_LIBRARIES_RELEASE
vulkan-1.lib
glfw3.lib
)
TARGET_LINK_LIBRARIES(${TARGET_NAME} debug "${ADDITIONAL_LIBRARIES_DEBUG}" optimized "${ADDITIONAL_LIBRARIES_RELEASE}")


add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory  
                ${PROJECT_SOURCE_DIR}/shaders 
                ${CMAKE_BINARY_DIR}/bin/shaders)
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory  
                ${PROJECT_SOURCE_DIR}/textures 
                ${CMAKE_BINARY_DIR}/bin/textures)
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory  
                ${PROJECT_SOURCE_DIR}/models 
                ${CMAKE_BINARY_DIR}/bin/models)