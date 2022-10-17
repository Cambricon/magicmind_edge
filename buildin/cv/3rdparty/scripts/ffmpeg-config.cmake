set(ffmpeg_path "${CMAKE_CURRENT_LIST_DIR}")

message(" ${CMAKE_CURRENT_LIST_DIR}")

set(FFMPEG_EXEC_DIR "${ffmpeg_path}/bin")
set(FFMPEG_LIBDIR "${ffmpeg_path}/lib")
set(FFMPEG_INCLUDE_DIRS "${ffmpeg_path}/include")

# library names
set(FFMPEG_LIBRARIES
    ${FFMPEG_LIBDIR}/libavformat.so
    ${FFMPEG_LIBDIR}/libavdevice.so
    ${FFMPEG_LIBDIR}/libavcodec.so
    ${FFMPEG_LIBDIR}/libavutil.so
    ${FFMPEG_LIBDIR}/libswscale.so
    ${FFMPEG_LIBDIR}/libswresample.so
    ${FFMPEG_LIBDIR}/libavfilter.so
    ${FFMPEG_LIBDIR}/libavresample.so
    pthread
)

# found status
set(FFMPEG_libavformat_FOUND TRUE)
set(FFMPEG_libavdevice_FOUND TRUE)
set(FFMPEG_libavcodec_FOUND TRUE)
set(FFMPEG_libavutil_FOUND TRUE)
set(FFMPEG_libswscale_FOUND TRUE)
set(FFMPEG_libswresample_FOUND TRUE)
set(FFMPEG_libavfilter_FOUND TRUE)
set(FFMPEG_libavresample_FOUND TRUE)

# library versions
# library versions, 注意这几个变量，一定要设置为全局CACHE变量
set(FFMPEG_libavutil_VERSION 56.31.100 CACHE INTERNAL "FFMPEG_libavutil_VERSION") # info
set(FFMPEG_libavcodec_VERSION 58.54.100 CACHE INTERNAL "FFMPEG_libavcodec_VERSION") # info
set(FFMPEG_libavformat_VERSION 58.29.100 CACHE INTERNAL "FFMPEG_libavformat_VERSION") # info
set(FFMPEG_libavdevice_VERSION 58.8.100 CACHE INTERNAL "FFMPEG_libavdevice_VERSION") # info
set(FFMPEG_libavfilter_VERSION 7.57.100 CACHE INTERNAL "FFMPEG_libavfilter_VERSION") # info
set(FFMPEG_libswscale_VERSION 5.5.100 CACHE INTERNAL "FFMPEG_libswscale_VERSION") # info
set(FFMPEG_libswresample_VERSION 3.5.100 CACHE INTERNAL "FFMPEG_libswresample_VERSION") # info

set(FFMPEG_FOUND TRUE)
set(FFMPEG_LIBS ${FFMPEG_LIBRARIES})
















# set(ffmpeg_path "${CMAKE_CURRENT_LIST_DIR}")

# message("ffmpeg_path: ${ffmpeg_path}")

# set(FFMPEG_EXEC_DIR "${ffmpeg_path}/bin")
# set(FFMPEG_LIBDIR "${ffmpeg_path}/lib")
# set(FFMPEG_INCLUDE_DIRS "${ffmpeg_path}/include")

# # library names
# set(FFMPEG_LIBRARIES
#     ${FFMPEG_LIBDIR}/libavformat.a
#     ${FFMPEG_LIBDIR}/libavdevice.a
#     ${FFMPEG_LIBDIR}/libavcodec.a
#     ${FFMPEG_LIBDIR}/libavutil.a
#     ${FFMPEG_LIBDIR}/libswscale.a
#     ${FFMPEG_LIBDIR}/libswresample.a
#     ${FFMPEG_LIBDIR}/libavfilter.a
# )

# # found status
# set(FFMPEG_libavformat_FOUND TRUE)
# set(FFMPEG_libavdevice_FOUND TRUE)
# set(FFMPEG_libavcodec_FOUND TRUE)
# set(FFMPEG_libavutil_FOUND TRUE)
# set(FFMPEG_libswscale_FOUND TRUE)
# set(FFMPEG_libswresample_FOUND TRUE)
# set(FFMPEG_libavfilter_FOUND TRUE)



# set(FFMPEG_FOUND TRUE)
# set(FFMPEG_LIBS ${FFMPEG_LIBRARIES})

# status("    #################################### FFMPEG:"       FFMPEG_FOUND         THEN "YES (find_package)"                       ELSE "NO (find_package)")
# status("      avcodec:"      FFMPEG_libavcodec_VERSION    THEN "YES (${FFMPEG_libavcodec_VERSION})"    ELSE NO)
# status("      avformat:"     FFMPEG_libavformat_VERSION   THEN "YES (${FFMPEG_libavformat_VERSION})"   ELSE NO)
# status("      avutil:"       FFMPEG_libavutil_VERSION     THEN "YES (${FFMPEG_libavutil_VERSION})"     ELSE NO)
# status("      swscale:"      FFMPEG_libswscale_VERSION    THEN "YES (${FFMPEG_libswscale_VERSION})"    ELSE NO)
# status("      avresample:"   FFMPEG_libavresample_VERSION THEN "YES (${FFMPEG_libavresample_VERSION})" ELSE NO)