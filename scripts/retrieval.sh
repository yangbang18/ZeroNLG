model=$1
outputPath=$2
clip=$3

model=${model:-zeronlg-4langs-vc}
outputPath=${outputPath:-}
clip=${clip:-}

python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset flickr30k --lang en
python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset flickr30k --lang zh
python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset flickr30k --lang de
python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset flickr30k --lang fr
python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset flickr30k --lang cs
python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset coco --lang en
python infer_retrieval.py --model ${model} --output_path "${outputPath}" --clip_model_name "${clip}" --dataset coco --lang ja
#python infer_retrieval.py --model ${model} --output_path "${outputPath}" --dataset msrvtt --lang en

# bash scripts/translate.sh zeronlg-4langs-vc output/zeronlg-4langs-vc
