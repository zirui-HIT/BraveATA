export API_KEY=YOUR_API_KEY
export BASE_URL=YOUR_BASE_URL
for MODEL in GPT-5.4; do
    for SCALE in nano; do
        for PART in theorem_elicitation autoformalization theorem_proving theorem_proving_formal; do
            DUMP_PATH=./generate/result/$MODEL/$SCALE
            if [ ! -d $DUMP_PATH ]; then
                mkdir -p $DUMP_PATH
            fi
            if [ -f $DUMP_PATH/generation.json ]; then
                DATA_FILE=$DUMP_PATH/generation.json
            else
                DATA_FILE=./BraveATA.json
            fi
            if [ "$SCALE" = "-" ]; then
                MODEL_NAME="$MODEL"
            else
                MODEL_NAME="$MODEL-$SCALE"
            fi

            python3 ./generate/inference.py \
            --data_file $DATA_FILE \
            --dump_file $DUMP_PATH/generation.json \
            --config_file ./generate/config/$MODEL.json \
            --llm_name_or_path $MODEL_NAME \
            --task $PART
        done
    done
done
