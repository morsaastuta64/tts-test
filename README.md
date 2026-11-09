El modelo LLM que se está utilizando es Qwen3-TTS. De momento está preparado para 2 acciones: clonar y finetunear voces.
- Para clonar voces, acceder a la carpeta "qwen/clone".
- Para finetunear voces, acceder a la carpeta "qwen/finetune".
La carpeta "library" tiene varios archivos de voz con sus transcripciones.

CLONE
Se pueden clonar voces a partir de un audio transcrito o a partir de una voz entrenada.
- El primer proceso se ejecuta con los scripts "clone-en.py" o "clone-sp.py" (sin argumentos), dependiendo de si se quiere clonar en inglés o en español. Requiere que se coloquen en la misma carpeta archivos "input.wav" e "input.txt"; el audio y su transcripción. También necesita un "output.txt" con el texto de salida. Se generarán 3 archivos "output_X.wav" con la voz recitando el texto de salida.
- El segundo proceso se ejecuta con el script "pretrained.py" (sin argumentos). Requiere igualmente un "output.txt" con el texto recitado, pero la entrada que toma es la información de una voz ya entrenada bajo el nombre de "speaker". (ESTE SCRIPT TODAVÍA NO HA PODIDO SER PROBADO)

FINETUNE
Esto sirve para entrenar una voz para a posteriori poder utilizarla para recitar textos. Tiene 2 pasos:
1. Ejecutar el siguiente comando: [python feed.py --device cuda:0 --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz --input_jsonl input.jsonl --output_jsonl output.jsonl]. Requiere un "input.jsonl" en el que se haga referencia a varios audios, sus transcripciones y un archivo común de sonido con la voz que se quiera utilizar para el entrenamiento. Esto produce un "output.jsonl".
2. Ejecutar el siguiente comando: [python train.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --output_model_path output --train_jsonl output.jsonl --batch_size 32 --lr 2e-6 --num_epochs 10 --speaker_name speaker]. Esto generará un archivo con la información de la voz entrenada bajo el nombre "speaker".