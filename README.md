# lorelei-speech-evaluation
The evaluation script for the LORELEI Situation Frames from Speech task.

The included python script will parse the ground truth and system output in the official LORELEI format and produce Precision-Recall (P-R) curves for the Relevance, Type and Type+Place evaluation layers.
The outputs include plots of the curves, as well as plain text recall-precision pairs for every sampled point.

It can be executed as: python3 -s [system output json] -g [ground truth directory] -o [output directory for the evaluation results].

For more details about the process followed to generate the metrics, see the included PDF file.

