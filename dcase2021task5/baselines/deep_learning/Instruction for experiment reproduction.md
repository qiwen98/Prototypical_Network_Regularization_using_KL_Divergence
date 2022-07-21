程序运行步骤：

第一步
download the developement set mentioned by dcase2021 task 5 and hable the directories in config.yaml file(make sure you do this steps before anything)

在目录下/home/wft/DCASE2021Task5/src/下进行如下命令行的执行：目录下运行如下命令行：
 1.  CUDA_VISIBLE_DEVICES=9  python main.py set.features=true
 2.  CUDA_VISIBLE_DEVICES=9 python main.py set.train=true
 3.  CUDA_VISIBLE_DEVICES=9 python main.py set.eval=true-----?这一步生成下一步要用的Eval_out_tim.csv文件。同时使用pre_best/目录下的weight文件：即model_best.pth.tar

 4.  命令模板：  python post_proc.py -val_path=/home/wft/dcaseTask5_with_40_f_measure/Development_Set/Validation_Set/ -evaluation_file=eval_output.csv -new_evaluation_file=new_eval_output.csv
     具体命令行：CUDA_VISIBLE_DEVICES=9  python post_proc.py -val_path=/home/wft/dcaseTask5_with_40_f_measure/Development_Set/Validation_Set/ -evaluation_file=Eval_out_tim.csv -new_evaluation_file=Eval_out_tim_new.csv
     把生成的文件 Eval_out_new.csv copy到如下第4/5 步骤中的Eval_out_new.csv进行 evaluation。

在目录 /home/wft/DCASE2021Task5/evaluation_metrics/下进行如下命令行的执行：

 5.  命令模板：python evaluation.py -pred_file=baseline_template_val_predictions.csv -ref_files_path=./Development_Set/Validation_Set/ -team_name=TESTteam -dataset=VAL -savepath=./
      具体命令行：CUDA_VISIBLE_DEVICES=9  python evaluation.py -pred_file=/home/wft/DCASE2021Task5/src/Eval_out_tim_new.csv -ref_files_path=/home/wft/dcaseTask5_with_40_f_measure/Development_Set/Validation_Set/ -team_name=TESTteam -dataset=VAL -savepath=./dict/

Steps of using the pretrianed model to reproduce our results.
1. CUDA_VISIBLE_DEVICES=9 python main.py set.eval=true-----?这一步生成下一步要用的Eval_out_tim.csv文件。同时使用pre_best/目录下的weight文件：即model_best.pth
2. python post_proc.py -val_path=/home/wft/dcaseTask5_with_40_f_measure/Development_Set/Validation_Set/ -evaluation_file=eval_output.csv -new_evaluation_file=new_eval_output.csv
3. python evaluation.py -pred_file=baseline_template_val_predictions.csv -ref_files_path=./Development_Set/Validation_Set/ -team_name=TESTteam -dataset=VAL -savepath=./