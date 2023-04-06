dataset=$1
lang=$2
model=$3
other=$4
outputRoot=$5

other=${other:-}
outputRoot=${outputRoot:-'output/train_caption'}

model=${model:-}
if [[ -z "$model" ]]; then
    folder=pt0_b32
else
    folder=pt1_b32

    # zero-shot
    python infer_caption.py \
    --auto \
    --model ${model} \
    --dataset ${dataset} \
    --lang ${lang} \
    --output_path ${outputRoot}/${folder}/${dataset}/${lang}/0% \
    --no_suffix_folder
fi

if [ $dataset = 'msrvtt' ]
then
    arr=(0.1% 1% 10%)
else
    arr=(0.01% 0.1% 1% 10%)    
fi

for r in "${arr[@]}"
do
    for n in {0..2}
    do
        subset="${r}_${n}"

        python train_caption.py \
        --warmup_steps 0 \
        --epochs 10 \
        --use_amp \
        --auto \
        --teacher_model_name clip-ViT-B-32 \
        --output_path ${outputRoot}/${folder}/${dataset}/${lang}/${subset} \
        --dataset $dataset \
        --lang $lang \
        --model "$model" \
        --subset $subset \
        $other
    done
done

# bash scripts/semi.sh coco en
# bash scripts/semi.sh msrvtt en
# bash scripts/semi.sh flickr30k de
# bash scripts/semi.sh flickr30k fr
# bash scripts/semi.sh vatex zh

# bash scripts/semi.sh coco en zeronlg-4langs-vc
# bash scripts/semi.sh msrvtt en zeronlg-4langs-vc
# bash scripts/semi.sh flickr30k de zeronlg-4langs-vc
# bash scripts/semi.sh flickr30k fr zeronlg-4langs-vc
# bash scripts/semi.sh vatex zh zeronlg-4langs-vc
