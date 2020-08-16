# AER

There are 2 main fiiles:
1. AER.py which conains the code of AER model.
2. hyper_param_eval.py which contains the code of optimization of hyper-parameters.

For running the code you need to open main.py file and to change the following pathes:
1. In line 170 to change the path to the folder which contains csv file for classification.

    evaluateFiles(True, 'min', YOUR_PATH)
2. In line 175 to change the path to the path of the output file.

    results_df.to_csv('YOUR_PATH\\results.csv', index=False)
