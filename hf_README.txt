changed_code/ 修改过的代码
figs/ 使用fast-bev自己的可视化代码，可视化的效果。
vis/ 使用bevdet的可视化代码，根据fast-bev的预测结果可视化的效果。
test_result/ 存储模型预测的结果。运行test.py时，需要指定jsonfile_prefix=$savepath。比如bevdet的例子，python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
