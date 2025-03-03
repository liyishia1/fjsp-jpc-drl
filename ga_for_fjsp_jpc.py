import time
import pandas as pd

from Params import env_configs
from data_dev.excel_to_data import getdata
from src.utils import parser, gantt
from src.genetic import encoding, decoding, genetic, termination
from src import config
from src.utils.util import *

env_configs.n_groups = 2


def ini_data():
    model_name = '{}-{}-{}-{}'.format(env_configs.n_groups, env_configs.n_ops, env_configs.n_mas,
                                      env_configs.n_jobs)

    # validation data set
    milp_data = pd.read_excel('data_dev/milpresult/{}.xlsx'.format(model_name), index_col=0)
    vali_data = getdata(model_name)
    vail_milp_result = milp_data['cmax'].to_numpy()
    return vali_data, vail_milp_result, model_name


def main(m,n, seed=20240705):
    np.random.seed(seed)
    random.seed(seed)
    vali_data, vail_milp_result, model_name = ini_data()
    maks = []
    gaps = []
    for i in range(m,n):
        ga_data = pd.DataFrame()
        vali_size_data = vali_data[i]
        machinesNb, jobs, jobs_prec = getjobs(*vali_size_data)
        jpc = tran_pre(jobs_prec)

        parameters = {'machinesNb': machinesNb,
                      'jobs': jobs,
                      'jpc': jpc}

        t0 = time.time()

        # Initialize the Population
        population = encoding.initializePopulation(parameters)
        gen = 1

        # Evaluate the population
        while not termination.shouldTerminate(population, gen):
            # Genetic Operators
            population, makespan = genetic.selection(population, parameters)
            ga_data['iter{}'.format(gen)] = makespan
            population = genetic.crossover(population, parameters)
            population = genetic.mutation(population, parameters)

            gen = gen + 1
            if gen%10 == 0:
                print(gen)

        sortedPop = sorted(population, key=lambda cpl: genetic.timeTaken(cpl, parameters))

        t1 = time.time()
        total_time = t1 - t0
        print("Finished in {0:.2f}s".format(total_time))

        # Termination Criteria Satisfied ?
        ga_data.to_excel('result/{}/{}/{}-{}.xlsx'.format(seed, model_name, model_name, i))
        makespan = genetic.timeTaken(sortedPop[0], parameters)
        gap = float(makespan) / float(vail_milp_result[i]) - 1
        maks.append(makespan)
        gaps.append(gap)
        gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, sortedPop[0][0], sortedPop[0][1]))

        tag1 = 'fig/{}/{}/{}-{}.pdf'.format(seed, model_name,model_name, i)

        filename = 'result/{}/{}/{}-{}.txt'.format(seed, model_name, model_name, i)
        with open(filename, 'w') as file:
            file.write(f"makespan: {makespan}\n")
            file.write(f"gap: {gap}\n")
        # tag2 = 'lat/{}/{}-{}.pdf'.format(seed, model_name, i)
        if config.latex_export:
            gantt.export_latex(gantt_data, tag=tag1)
        else:
            gantt.draw_chart(gantt_data, tag=tag1)

    df = pd.DataFrame({'makespan': maks, 'gap': gaps})
    df.to_excel('result/{}/{}-final.xlsx'.format(seed, model_name))



if __name__ == '__main__':
    st = time.time()
    main(0, 20)
    print(time.time() - st)
