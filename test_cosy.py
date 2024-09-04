from mlxs2s.cosyvoice_models import TransformerLM

MODEL_DIR = "/Users/liwei15/repo/CosyVoice/pretrained_models/CosyVoice-300M/"

TEXT_ENCODER_INPUT_SIZE = 512
LLM_INPUT_SIZE = 1024
LLM_OUTPUT_SIZE = 1024
SPK_EMBED_DIM = 192

llm_model = TransformerLM(
            text_encoder_input_size=TEXT_ENCODER_INPUT_SIZE,
            llm_input_size=LLM_INPUT_SIZE,
            llm_output_size=LLM_OUTPUT_SIZE,
            text_token_size=51866,
            speech_token_size=4096,
            text_encoder=ConformerEncoder(
                    input_size=TEXT_ENCODER_INPUT_SIZE,
                    output_size=1024,
                    attention_heads=16,
                    linear_units=4096,
                    num_blocks=6,
                    dropout_rate=0.1,
                    positional_dropout_rate=0.1,
                    attention_dropout_rate=0.0,
                    normalize_before=True,
                    input_layer='linear',
                    pos_enc_layer_type='rel_pos_espnet',
                    selfattention_layer_type='rel_selfattn',
                    use_cnn_module=False,
                    macaron_style=False,
                    use_dynamic_chunk=False,
                    use_dynamic_left_chunk=False,
                    static_chunk_size=1,
                ),
            llm=TransformerEncoder(
                    input_size=LLM_INPUT_SIZE,
                    output_size=LLM_OUTPUT_SIZE,
                    attention_heads=16,
                    linear_units=4096,
                    num_blocks=14,
                    dropout_rate=0.1,
                    positional_dropout_rate=0.1,
                    attention_dropout_rate=0.0,
                    input_layer='linear_legacy',
                    pos_enc_layer_type='rel_pos_espnet',
                    selfattention_layer_type='rel_selfattn',
                    static_chunk_size=1,
            ),
            length_normalized_loss=True,
            lsm_weight=0,
            spk_embed_dim=SPK_EMBED_DIM
)

llm_weight = torch.load(str(Path(MODEL_DIR) / 'llm.pt'), map_location='cpu')
llm_input = torch.load('llm_input.pt', map_location='cpu')
llm_model.load_state_dict(llm_weight)

# llm_input = llm_input.to('mps')
# for k in llm_input:
#     llm_input[k] = llm_input[k].to('mps')
# llm_model = llm_model
llm_model.eval()
# llm_model = llm_model.to('mps')
tts_speech_token = llm_model.inference(
    text=llm_input['text'],
    text_len=llm_input['text_len'],
    prompt_text=llm_input['prompt_text'],
    prompt_text_len=llm_input['prompt_text_len'],
    prompt_speech_token=llm_input['llm_prompt_speech_token'],
    prompt_speech_token_len=llm_input['llm_prompt_speech_token_len'],
    embedding=llm_input['llm_embedding'],
    beam_size=1,
    sampling=25,
    max_token_text_ratio=30,
    min_token_text_ratio=3
)
# cosy_model.load(
#     '{}/llm.pt'.format(MODEL_DIR),
#     '{}/flow.pt'.format(MODEL_DIR),
#     '{}/hift.pt'.format(MODEL_DIR)
# )
