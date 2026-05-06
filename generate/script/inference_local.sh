for MODEL in Qwen3.5; do
    for SCALE in 35B; do
        for PART in theorem_elicitation; do
            DUMP_PATH=./generate/result/$MODEL/$SCALE
            if [ ! -d ./model/$MODEL/$SCALE ]; then
                continue
            fi
            if [ ! -d $DUMP_PATH ]; then
                mkdir -p $DUMP_PATH
            fi
            if [ -f $DUMP_PATH/generation.json ]; then
                DATA_FILE=$DUMP_PATH/generation.json
            else
                DATA_FILE=./BraveATA.json
            fi

            python3 ./generate/inference.py \
            --data_file $DATA_FILE \
            --dump_file $DUMP_PATH/generation.json \
            --config_file ./generate/config/$MODEL.json \
            --llm_name_or_path ./model/$MODEL/$SCALE \
            --task $PART
        done
    done
done
