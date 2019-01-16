#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include <obs-module.h>
#include <media-io/audio-math.h>
#include <util/platform.h>
#include <util/circlebuf.h>
#include <util/threading.h>

/* -------------------------------------------------------- */

#define do_log(level, format, ...) \
	blog(level, "[expander: '%s'] " format, \
			obs_source_get_name(cd->context), ##__VA_ARGS__)

#define warn(format, ...)  do_log(LOG_WARNING, format, ##__VA_ARGS__)
#define info(format, ...)  do_log(LOG_INFO,    format, ##__VA_ARGS__)

#ifdef _DEBUG
#define debug(format, ...) do_log(LOG_DEBUG,   format, ##__VA_ARGS__)
#else
#define debug(format, ...)
#endif

/* -------------------------------------------------------- */

#define S_RATIO                         "ratio"
#define S_THRESHOLD                     "threshold"
#define S_ATTACK_TIME                   "attack_time"
#define S_RELEASE_TIME                  "release_time"
#define S_OUTPUT_GAIN                   "output_gain"
#define S_DETECTOR                      "detector"
#define S_PRESETS                       "presets"

#define MT_ obs_module_text
#define TEXT_RATIO                      MT_("expander.Ratio")
#define TEXT_THRESHOLD                  MT_("expander.Threshold")
#define TEXT_ATTACK_TIME                MT_("expander.AttackTime")
#define TEXT_RELEASE_TIME               MT_("expander.ReleaseTime")
#define TEXT_OUTPUT_GAIN                MT_("expander.OutputGain")
#define TEXT_DETECTOR                   MT_("expander.Detector")
#define TEXT_PEAK                       MT_("expander.Peak")
#define TEXT_RMS                        MT_("expander.RMS")
#define TEXT_RMS_STILLWELL              MT_("expander.RMS.Stillwell")
#define TEXT_NONE                       MT_("expander.None")
#define TEXT_PRESETS                    MT_("expander.Presets")
#define TEXT_PRESETS_EXP                MT_("expander.Presets.Expander")
#define TEXT_PRESETS_GATE               MT_("expander.Presets.Gate")

#define MIN_RATIO                       1.0
#define MAX_RATIO                       20.0
#define MIN_THRESHOLD_DB                -60.0
#define MAX_THRESHOLD_DB                0.0f
#define MIN_OUTPUT_GAIN_DB              -32.0
#define MAX_OUTPUT_GAIN_DB              32.0
#define MIN_ATK_RLS_MS                  1
#define MAX_RLS_MS                      1000
#define MAX_ATK_MS                      100
#define DEFAULT_AUDIO_BUF_MS            10

#define MS_IN_S                         1000
#define MS_IN_S_F                       ((float)MS_IN_S)

/* -------------------------------------------------------- */

struct expander_data {
	obs_source_t *context;
	float *envelope_buf;
	size_t envelope_buf_len;

	float ratio;
	float threshold;
	float attack_gain;
	float release_gain;
	float output_gain;

	size_t num_channels;
	size_t sample_rate;
	float envelope;
	float slope;
	int  detector;
	float runave;
	bool is_gate;
};

enum {
	RMS_DETECT,
	RMS_STILLWELL_DETECT,
	PEAK_DETECT,
	NO_DETECT,
};
/* -------------------------------------------------------- */

static void resize_env_buffer(struct expander_data *cd, size_t len)
{
	cd->envelope_buf_len = len;
	cd->envelope_buf = brealloc(cd->envelope_buf, len * sizeof(float));
}

static inline float gain_coefficient(uint32_t sample_rate, float time)
{
	return (float)exp(-1.0f / (sample_rate * time));
}

static const char *expander_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Expander");
}


static void expander_defaults(obs_data_t *s)
{
	const char *presets = obs_data_get_string(s, S_PRESETS);
	bool is_expander_preset = true;
	if (strcmp(presets, "gate") == 0)
		is_expander_preset = false;
	obs_data_set_default_string(s, S_PRESETS, is_expander_preset ? "expander" : "gate");
	obs_data_set_default_double(s, S_RATIO, is_expander_preset ? 2.0 : 10.0);
	obs_data_set_default_double(s, S_THRESHOLD, -40.0f);
	obs_data_set_default_int(s, S_ATTACK_TIME, 10);
	obs_data_set_default_int(s, S_RELEASE_TIME, is_expander_preset ? 50 : 125);
	obs_data_set_default_double(s, S_OUTPUT_GAIN, 0.0);
	obs_data_set_default_string(s, S_DETECTOR, "RMS");
}

static void expander_update(void *data, obs_data_t *s)
{
	struct expander_data *cd = data;
	const char *presets = obs_data_get_string(s, S_PRESETS);
	if (strcmp(presets, "expander") == 0 && cd->is_gate) {
		obs_data_clear(s);
		obs_data_set_string(s, S_PRESETS, "expander");
		expander_defaults(s);
		cd->is_gate = false;
	}
	if (strcmp(presets, "gate") == 0 && !cd->is_gate) {
		obs_data_clear(s);
		obs_data_set_string(s, S_PRESETS, "gate");
		expander_defaults(s);
		cd->is_gate = true;
	}

	const uint32_t sample_rate =
			audio_output_get_sample_rate(obs_get_audio());
	const size_t num_channels =
			audio_output_get_channels(obs_get_audio());
	const float attack_time_ms =
			(float)obs_data_get_int(s, S_ATTACK_TIME);
	const float release_time_ms =
			(float)obs_data_get_int(s, S_RELEASE_TIME);
	const float output_gain_db =
			(float)obs_data_get_double(s, S_OUTPUT_GAIN);

	cd->ratio = (float)obs_data_get_double(s, S_RATIO);
	cd->threshold = (float)obs_data_get_double(s, S_THRESHOLD);
	cd->attack_gain = gain_coefficient(sample_rate,
			attack_time_ms / MS_IN_S_F);
	cd->release_gain = gain_coefficient(sample_rate,
			release_time_ms / MS_IN_S_F);
	cd->output_gain = db_to_mul(output_gain_db);
	cd->num_channels = num_channels;
	cd->sample_rate = sample_rate;
	cd->slope = 1.0f - cd->ratio;

	const char *detect_mode = obs_data_get_string(s, S_DETECTOR);
	if (strcmp(detect_mode, "RMS") == 0)
		cd->detector = RMS_DETECT;
	if (strcmp(detect_mode, "RMS Stillwell") == 0)
		cd->detector = RMS_STILLWELL_DETECT;
	if (strcmp(detect_mode, "peak") == 0)
		cd->detector = PEAK_DETECT;
	if (strcmp(detect_mode, "none") == 0)
		cd->detector = NO_DETECT;

	size_t sample_len = sample_rate * DEFAULT_AUDIO_BUF_MS / MS_IN_S;
	if (cd->envelope_buf_len == 0)
		resize_env_buffer(cd, sample_len);
}

static void *expander_create(obs_data_t *settings, obs_source_t *filter)
{
	struct expander_data *cd = bzalloc(sizeof(struct expander_data));
	cd->context = filter;
	cd->runave = 0;
	cd->is_gate = false;
	const char *presets = obs_data_get_string(settings, S_PRESETS);
	if (strcmp(presets, "gate") == 0)
		cd->is_gate = true;

	expander_update(cd, settings);
	return cd;
}

static void expander_destroy(void *data)
{
	struct expander_data *cd = data;

	bfree(cd->envelope_buf);
	bfree(cd);
}

static void analyze_envelope(struct expander_data *cd,
	float **samples, const uint32_t num_samples)
{
	if (cd->envelope_buf_len < num_samples) {
		resize_env_buffer(cd, num_samples);
	}

	const float attack_gain = cd->attack_gain;
	const float release_gain = cd->release_gain;

	const float rmscoef = exp2f((float)-100.0 / (float)cd->sample_rate); // 10 ms RMS window
	const float peakcoef = exp2f((float)-1000.0 /((float)0.0025 * (float)cd->sample_rate)); // 2.5 microsec Peak window

	float *runave = bmalloc(num_samples * sizeof(float));
	float *maxspl = bmalloc(num_samples * sizeof(float));
	float *env_in = bmalloc(num_samples * sizeof(float));

	memset(cd->envelope_buf, 0, num_samples * sizeof(cd->envelope_buf[0]));
	for (size_t chan = 0; chan < cd->num_channels; ++chan) {
		if (!samples[chan])
			continue;

		float *envelope_buf = cd->envelope_buf;
		float env = cd->envelope;
		float RMSdet = 0;

		runave[0] = cd->runave;
		maxspl[0] = fabsf(samples[chan][0]);
		env_in[0] = sqrtf(fmaxf(cd->runave, 0));

		for (uint32_t i = 1; i < num_samples; ++i) {
			if (cd->detector == RMS_DETECT) {
				runave[i] = rmscoef * runave[i - 1] + (1 - rmscoef) * powf(samples[chan][i], 2);
			}
			if (cd->detector == RMS_STILLWELL_DETECT) {
				runave[i] = 2.08136898 * rmscoef * runave[i - 1] + (1 - rmscoef) * powf(samples[chan][i], 2);
			}
			if (cd->detector == PEAK_DETECT) {
				maxspl[i] = powf(fmaxf(fabsf(maxspl[i-1]), fabsf(samples[chan][i])), 2);
				runave[i] = peakcoef * runave[i - 1] + (1- peakcoef) * maxspl[i];
			}
			if (cd->detector == NO_DETECT) {
				runave[i] = powf(samples[chan][i], 2);
			}
			env_in[i] = sqrtf(fmaxf(runave[i], 0));
		}
		cd->runave = runave[num_samples-1];

		for (uint32_t i = 0; i < num_samples; ++i) {

			if (env < env_in[i]) {
				env = env_in[i] + attack_gain * (env - env_in[i]);
			} else {
				env = env_in[i] + release_gain * (env - env_in[i]);
			}
			envelope_buf[i] = fmaxf(envelope_buf[i], env);
		}

	}
	cd->envelope = cd->envelope_buf[num_samples - 1];
	bfree(runave);
	bfree(maxspl);
	bfree(env_in);
}

static inline void process_expansion(const struct expander_data *cd,
	float **samples, uint32_t num_samples)
{
	for (size_t i = 0; i < num_samples; ++i) {
		const float env_db = mul_to_db(cd->envelope_buf[i]);
		float gain = fmaxf(cd->slope * (cd->threshold - env_db), -60.0);
		gain = db_to_mul(fminf(0, gain));

		for (size_t c = 0; c < cd->num_channels; ++c) {
			if (samples[c]) {
				samples[c][i] *= gain * cd->output_gain;
			}
		}
	}
}

static struct obs_audio_data *expander_filter_audio(void *data,
	struct obs_audio_data *audio)
{
	struct expander_data *cd = data;

	const uint32_t num_samples = audio->frames;
	if (num_samples == 0)
		return audio;

	float **samples = (float**)audio->data;

	analyze_envelope(cd, samples, num_samples);
	process_expansion(cd, samples, num_samples);
	return audio;
}

static bool presets_changed(obs_properties_t *props, obs_property_t *prop,
	obs_data_t *settings)
{
	return true;
}

static obs_properties_t *expander_properties(void *data)
{
	struct expander_data *cd = data;
	obs_properties_t *props = obs_properties_create();
	obs_source_t *parent = NULL;

	if (cd)
		parent = obs_filter_get_parent(cd->context);

	obs_property_t *presets = obs_properties_add_list(props, S_PRESETS,
			TEXT_PRESETS, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(presets, TEXT_PRESETS_EXP, "expander");
	obs_property_list_add_string(presets, TEXT_PRESETS_GATE, "gate");
	obs_property_set_modified_callback(presets, presets_changed);
	obs_properties_add_float_slider(props, S_RATIO,
			TEXT_RATIO, MIN_RATIO, MAX_RATIO, 0.1);
	obs_properties_add_float_slider(props, S_THRESHOLD,
			TEXT_THRESHOLD, MIN_THRESHOLD_DB, MAX_THRESHOLD_DB, 0.1);
	obs_properties_add_int_slider(props, S_ATTACK_TIME,
			TEXT_ATTACK_TIME, MIN_ATK_RLS_MS, MAX_ATK_MS, 1);
	obs_properties_add_int_slider(props, S_RELEASE_TIME,
			TEXT_RELEASE_TIME, MIN_ATK_RLS_MS, MAX_RLS_MS, 1);
	obs_properties_add_float_slider(props, S_OUTPUT_GAIN,
			TEXT_OUTPUT_GAIN, MIN_OUTPUT_GAIN_DB, MAX_OUTPUT_GAIN_DB, 0.1);
	obs_property_t *detect = obs_properties_add_list(props, S_DETECTOR,
			TEXT_DETECTOR, OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(detect, TEXT_RMS, "RMS");
	obs_property_list_add_string(detect, TEXT_RMS_STILLWELL, "RMS Stillwell");
	obs_property_list_add_string(detect, TEXT_PEAK, "peak");
	obs_property_list_add_string(detect, TEXT_NONE, "none");

	UNUSED_PARAMETER(data);
	return props;
}

struct obs_source_info expander_filter = {
	.id = "expander_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_AUDIO,
	.get_name = expander_name,
	.create = expander_create,
	.destroy = expander_destroy,
	.update = expander_update,
	.filter_audio = expander_filter_audio,
	.get_defaults = expander_defaults,
	.get_properties = expander_properties,
};
