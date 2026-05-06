for MODEL in Qwen3.5; do
    for SCALE in 397B; do
        for PART in theorem_elicitation theorem_proving; do
            DATA_FILE=./generate/result/$MODEL/$SCALE/generation.json
            DUMP_FILE=./generate/result/$MODEL/$SCALE/generation.eval.json
            if [ ! -f $DATA_FILE ]; then
                continue
            fi
            if [ -f $DUMP_FILE ]; then
                DATA_FILE=$DUMP_FILE
            fi

            python3 ./generate/evaluate.py \
            --data_file $DATA_FILE \
            --dump_file $DUMP_FILE \
            --task $PART
        done
    done
done
