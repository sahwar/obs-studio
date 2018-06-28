#include "audio-repack.h"

#include <emmintrin.h>

int check_buffer(struct audio_repack *repack,
		uint32_t frame_count)
{
	const uint32_t new_size = frame_count * repack->base_dst_size;

	if (repack->packet_size < new_size) {
		repack->packet_buffer = brealloc(
				repack->packet_buffer, new_size);
		if (!repack->packet_buffer)
			return -1;

		repack->packet_size = new_size;
	}

	return 0;
}

/*
 * Squash arrays; for instance:
 * 2.1:
 *
 * | FL | FR | LFE | emp | emp | emp |emp |emp |
 * |    |    |
 * | FL | FR | LFE |
 */

int repack_squash(struct audio_repack *repack,
		const uint8_t *bsrc, uint32_t frame_count)
{
	if (check_buffer(repack, frame_count) < 0)
		return -1;

	int squash = repack->extra_dst_size;
	const __m128i *src = (__m128i *)bsrc;
	const __m128i *esrc = src + 2 * frame_count;
	uint32_t *dst = (uint32_t *)repack->packet_buffer;
	/* 2.1 audio does not require re-ordering but still needs squashing
	 * in order to avoid sampling issues.
	 */
	if (repack->base_src_size == 32) {
		while (src != esrc) {
			__m128i target_l = _mm_load_si128(src++);
			__m128i target_h = _mm_load_si128(src++);
			_mm_storeu_si128((__m128i *)dst, target_l);
			dst += 4;
			if (squash < 4)
				_mm_storeu_si128((__m128i *)dst, target_h);
			dst += 4 - squash;
		}
	}
	/* Squash empty channels for 9 to 15 channels.
	 */
	if (repack->base_src_size == 64) {
		esrc = src + 4 *frame_count;
		while (src != esrc) {
			__m128i target_l = _mm_load_si128(src++);
			__m128i target_h = _mm_load_si128(src++);
			__m128i target2_l = _mm_load_si128(src++);
			__m128i target2_h = _mm_load_si128(src++);
			_mm_storeu_si128((__m128i *)dst, target_l);
			dst += 4;
			_mm_storeu_si128((__m128i *)dst, target_h);
			dst += 4;
			_mm_storeu_si128((__m128i *)dst, target2_l);
			dst += 4;
			if (squash < 4)
				_mm_storeu_si128((__m128i *)dst, target2_h);
			dst += 4 - squash;
		}
	}

	return 0;
}

int audio_repack_init(struct audio_repack *repack,
		audio_repack_mode_t repack_mode, uint8_t sample_bit)
{
	memset(repack, 0, sizeof(*repack));

	if (sample_bit != 32)
		return -1;

	repack->base_dst_size = (int)repack_mode * (sample_bit / 8);
	repack->repack_func = &repack_squash;

	if (repack_mode <= 8) {
		repack->base_src_size = 8 * (sample_bit / 8);
		repack->extra_dst_size = 8 - (int)repack_mode;
	}

	if (repack_mode > 8) {
		repack->base_src_size = 16 * (sample_bit / 8);
		repack->extra_dst_size = 16 - (int)repack_mode;
	}

	return 0;
}

void audio_repack_free(struct audio_repack *repack)
{
	if (repack->packet_buffer)
		bfree(repack->packet_buffer);

	memset(repack, 0, sizeof(*repack));
}
