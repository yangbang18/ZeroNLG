model=$1
outputPath=$2
outputPaht=${outputPath:-}

python infer_translate.py --model ${model} --output_path "${outputPath}" --dataset flickr30k --source en --target zh
python infer_translate.py --model ${model} --output_path "${outputPath}" --dataset flickr30k --source en --target de
python infer_translate.py --model ${model} --output_path "${outputPath}" --dataset flickr30k --source en --target fr
python infer_translate.py --model ${model} --output_path "${outputPath}" --dataset flickr30k --source zh --target de
python infer_translate.py --model ${model} --output_path "${outputPath}" --dataset flickr30k --source zh --target fr
python infer_translate.py --model ${model} --output_path "${outputPath}" --dataset flickr30k --source de --target fr

# bash scripts/translate.sh zeronlg-4langs-mt output/zeronlg-4langs-mt
