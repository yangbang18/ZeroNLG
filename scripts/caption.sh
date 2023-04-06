model=$1
outputPath=$2
outputPaht=${outputPath:-}

python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset coco --lang en
#python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang en
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang zh
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang de
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang fr
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset msrvtt --lang en
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset vatex --lang zh
#python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset vatex --lang en

# bash scripts/caption.sh zeronlg-4langs-vc output/zeronlg-4langs-vc
