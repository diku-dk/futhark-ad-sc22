#!/usr/bin/env python3

ba_file = 'ADBench/tmp/Release/ba/{}/ba5_n257_m65132_p225911_times_{}.txt'
lstm_file = 'ADBench/tmp/Release/lstm/{}/lstm_l4_c4096_times_{}.txt'
gmm_file = 'ADBench/tmp/Release/gmm/10k/{}/gmm_d64_K200_times_{}.txt'
hand_complicated_file = 'ADBench/tmp/Release/hand/complicated_big/{}/hand5_t26_c800_times_{}.txt'
hand_simple_file = 'ADBench/tmp/Release/hand/simple_big/{}/hand5_t26_c800_times_{}.txt'

def file_for_tool(template, tool):
    return template.format(tool, tool)

def runtime_results(template, tool):
    obj,J = open(file_for_tool(template, tool)).read().split()
    return (float(obj), float(J))

def J_overhead(template, tool):
    obj,J = runtime_results(template, tool)
    return J/obj

def table_row(tool):
    print('%8s | %4.1fx    %4.1fx   %4.1fx   %7.1fx    %6.1fx'
      % (tool,
         J_overhead(ba_file, tool),
         J_overhead(lstm_file, tool),
         J_overhead(gmm_file, tool),
         J_overhead(hand_complicated_file, tool),
         J_overhead(hand_simple_file, tool)))

print('Tool     |    BA   D-LSTM     GMM            HAND')
print('         |                             Comp.     Simple')
print('---------+---------------------------------------------')
table_row('Futhark')
table_row('Tapenade')
table_row('Manual')


