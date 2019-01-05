#pragma once

static inline int64_t rescale_ts(int64_t val, AVCodecContext *context,
		AVRational new_base)
{
	return av_rescale_q_rnd(val, context->time_base, new_base,
			AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
}

static inline enum AVPixelFormat obs_to_ffmpeg_video_format(
		enum video_format format)
{
	switch (format) {
	case VIDEO_FORMAT_NONE: return AV_PIX_FMT_NONE;
	case VIDEO_FORMAT_I444: return AV_PIX_FMT_YUV444P;
	case VIDEO_FORMAT_I420: return AV_PIX_FMT_YUV420P;
	case VIDEO_FORMAT_NV12: return AV_PIX_FMT_NV12;
	case VIDEO_FORMAT_YVYU: return AV_PIX_FMT_NONE;
	case VIDEO_FORMAT_YUY2: return AV_PIX_FMT_YUYV422;
	case VIDEO_FORMAT_UYVY: return AV_PIX_FMT_UYVY422;
	case VIDEO_FORMAT_RGBA: return AV_PIX_FMT_RGBA;
	case VIDEO_FORMAT_BGRA: return AV_PIX_FMT_BGRA;
	case VIDEO_FORMAT_BGRX: return AV_PIX_FMT_BGRA;
	case VIDEO_FORMAT_Y800: return AV_PIX_FMT_GRAY8;
	}

	return AV_PIX_FMT_NONE;
}

static inline enum video_format ffmpeg_to_obs_video_format(
		enum AVPixelFormat format)
{
	switch (format) {
	case AV_PIX_FMT_YUV444P: return VIDEO_FORMAT_I444;
	case AV_PIX_FMT_YUV420P: return VIDEO_FORMAT_I420;
	case AV_PIX_FMT_NV12:    return VIDEO_FORMAT_NV12;
	case AV_PIX_FMT_YUYV422: return VIDEO_FORMAT_YUY2;
	case AV_PIX_FMT_UYVY422: return VIDEO_FORMAT_UYVY;
	case AV_PIX_FMT_RGBA:    return VIDEO_FORMAT_RGBA;
	case AV_PIX_FMT_BGRA:    return VIDEO_FORMAT_BGRA;
	case AV_PIX_FMT_GRAY8:   return VIDEO_FORMAT_Y800;
	case AV_PIX_FMT_NONE:
	default:                 return VIDEO_FORMAT_NONE;
	}
}

static inline uint64_t convert_speaker_layout(enum speaker_layout layout)
{
	switch (layout) {
	case SPEAKERS_UNKNOWN:          return 0;
	case SPEAKERS_MONO:             return AV_CH_LAYOUT_MONO;
	case SPEAKERS_STEREO:           return AV_CH_LAYOUT_STEREO;
	case SPEAKERS_2POINT1:          return AV_CH_LAYOUT_SURROUND;
	case SPEAKERS_3POINT0:          return AV_CH_LAYOUT_SURROUND;
	case SPEAKERS_4POINT0:          return AV_CH_LAYOUT_4POINT0;
	case SPEAKERS_QUAD:             return AV_CH_LAYOUT_QUAD;
	case SPEAKERS_3POINT1:          return AV_CH_LAYOUT_3POINT1;
	case SPEAKERS_5POINT0:          return AV_CH_LAYOUT_5POINT0_BACK;
	case SPEAKERS_4POINT1:          return AV_CH_LAYOUT_4POINT1;
	case SPEAKERS_5POINT1:          return AV_CH_LAYOUT_5POINT1_BACK;
	case SPEAKERS_6POINT0:          return AV_CH_LAYOUT_6POINT0;
	case SPEAKERS_6POINT1:          return AV_CH_LAYOUT_6POINT1;
	case SPEAKERS_7POINT0:          return AV_CH_LAYOUT_7POINT0;
	case SPEAKERS_7POINT1:          return AV_CH_LAYOUT_7POINT1;
	case SPEAKERS_OCTAGONAL:        return AV_CH_LAYOUT_OCTAGONAL;
	case SPEAKERS_9POINT0:          return (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER);
	case SPEAKERS_10POINT0:         return (AV_CH_LAYOUT_6POINT0_FRONT | AV_CH_BACK_CENTER | AV_CH_BACK_LEFT | AV_CH_BACK_RIGHT | AV_CH_TOP_CENTER);
	case SPEAKERS_11POINT0:         return (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT);
	case SPEAKERS_12POINT0:         return (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER);
	case SPEAKERS_13POINT0:         return (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER | AV_CH_TOP_BACK_CENTER);
	case SPEAKERS_14POINT0:         return (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER | AV_CH_TOP_BACK_LEFT | AV_CH_TOP_BACK_RIGHT);
	case SPEAKERS_15POINT0:         return (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER | AV_CH_TOP_BACK_CENTER | AV_CH_TOP_BACK_LEFT | AV_CH_TOP_BACK_RIGHT);
	case SPEAKERS_HEXADECAGONAL:    return AV_CH_LAYOUT_HEXADECAGONAL;
	}

	/* shouldn't get here */
	return 0;
}

static inline enum speaker_layout convert_ff_channel_layout(uint64_t  channel_layout)
{
	switch (channel_layout) {
	case AV_CH_LAYOUT_MONO:                           return SPEAKERS_MONO;
	case AV_CH_LAYOUT_STEREO:                         return SPEAKERS_STEREO;
	case AV_CH_LAYOUT_2POINT1:                        return SPEAKERS_2POINT1;
	case AV_CH_LAYOUT_2_1:
	case AV_CH_LAYOUT_SURROUND:                       return SPEAKERS_3POINT0;
	case AV_CH_LAYOUT_3POINT1:                        return SPEAKERS_3POINT1;
	case AV_CH_LAYOUT_2_2:
	case AV_CH_LAYOUT_4POINT0:                        return SPEAKERS_4POINT0;
	case AV_CH_LAYOUT_QUAD:                           return SPEAKERS_QUAD;
	case AV_CH_LAYOUT_4POINT1:                        return SPEAKERS_4POINT1;
	case AV_CH_LAYOUT_5POINT0:
	case AV_CH_LAYOUT_5POINT0_BACK:                   return SPEAKERS_5POINT0;
	case AV_CH_LAYOUT_5POINT1:
	case AV_CH_LAYOUT_5POINT1_BACK:                   return SPEAKERS_5POINT1;
	case AV_CH_LAYOUT_6POINT0:
	case AV_CH_LAYOUT_6POINT0_FRONT:
	case AV_CH_LAYOUT_HEXAGONAL:                      return SPEAKERS_6POINT0;
	case AV_CH_LAYOUT_6POINT1:
	case AV_CH_LAYOUT_6POINT1_BACK:
	case AV_CH_LAYOUT_6POINT1_FRONT:	          return SPEAKERS_6POINT1;
	case AV_CH_LAYOUT_7POINT0:
	case AV_CH_LAYOUT_7POINT0_FRONT:	          return SPEAKERS_7POINT0;
	case AV_CH_LAYOUT_7POINT1:
	case AV_CH_LAYOUT_7POINT1_WIDE:
	case AV_CH_LAYOUT_7POINT1_WIDE_BACK:              return SPEAKERS_7POINT1;
	case AV_CH_LAYOUT_OCTAGONAL:                      return SPEAKERS_OCTAGONAL;
	case (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER): return SPEAKERS_9POINT0;
	case (AV_CH_LAYOUT_6POINT0_FRONT | AV_CH_BACK_CENTER | AV_CH_BACK_LEFT | AV_CH_BACK_RIGHT | AV_CH_TOP_CENTER):
	                                                  return SPEAKERS_10POINT0;
	case (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT):
	                                                  return SPEAKERS_11POINT0;
	case (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER):
							  return SPEAKERS_12POINT0;
	case (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER | AV_CH_TOP_BACK_CENTER):
							  return SPEAKERS_13POINT0;
	case (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER | AV_CH_TOP_BACK_LEFT | AV_CH_TOP_BACK_RIGHT):
							  return SPEAKERS_14POINT0;
	case (AV_CH_LAYOUT_OCTAGONAL | AV_CH_TOP_CENTER | AV_CH_TOP_FRONT_LEFT | AV_CH_TOP_FRONT_RIGHT | AV_CH_TOP_FRONT_CENTER | AV_CH_TOP_BACK_CENTER | AV_CH_TOP_BACK_LEFT | AV_CH_TOP_BACK_RIGHT):
							  return SPEAKERS_15POINT0;
	case AV_CH_LAYOUT_HEXADECAGONAL:                  return SPEAKERS_HEXADECAGONAL;
	}
	/* shouldn't get here */
	return  SPEAKERS_UNKNOWN;
}

static inline enum audio_format convert_ffmpeg_sample_format(
		enum AVSampleFormat format)
{
	switch ((uint32_t)format) {
	case AV_SAMPLE_FMT_U8:   return AUDIO_FORMAT_U8BIT;
	case AV_SAMPLE_FMT_S16:  return AUDIO_FORMAT_16BIT;
	case AV_SAMPLE_FMT_S32:  return AUDIO_FORMAT_32BIT;
	case AV_SAMPLE_FMT_FLT:  return AUDIO_FORMAT_FLOAT;
	case AV_SAMPLE_FMT_U8P:  return AUDIO_FORMAT_U8BIT_PLANAR;
	case AV_SAMPLE_FMT_S16P: return AUDIO_FORMAT_16BIT_PLANAR;
	case AV_SAMPLE_FMT_S32P: return AUDIO_FORMAT_32BIT_PLANAR;
	case AV_SAMPLE_FMT_FLTP: return AUDIO_FORMAT_FLOAT_PLANAR;
	}

	/* shouldn't get here */
	return AUDIO_FORMAT_16BIT;
}
