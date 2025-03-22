import random
import numpy as np
import gradio as gr
import pandas as pd
from deap import base, creator, tools

# 全局变量存储优化状态
state = {
    "toolbox": None,
    "population": None,
    "current_idx": 0,
    "current_gen": 0,
    "history": [],
    "best_ind": None,
    "running": False,
    "params": {}
}


def create_individual(min_max):
    return [
        random.uniform(min_max["kp_min"], min_max["kp_max"]),
        random.uniform(min_max["ki_min"], min_max["ki_max"]),
        random.uniform(min_max["kd_min"], min_max["kd_max"])
    ]


def init_toolbox(min_max):
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_individual(min_max))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def start_optimization(kp_min, kp_max, ki_min, ki_max, kd_min, kd_max,
                       pop_size, max_gen, cx_prob, mut_prob):
    state.update({
        "params": {
            "kp_range": (kp_min, kp_max),
            "ki_range": (ki_min, ki_max),
            "kd_range": (kd_min, kd_max),
            "pop_size": pop_size,
            "max_gen": max_gen,
            "cx_prob": cx_prob,
            "mut_prob": mut_prob
        },
        "current_gen": 0,
        "history": [],
        "best_ind": None,
        "running": True
    })

    min_max = {
        "kp_min": kp_min, "kp_max": kp_max,
        "ki_min": ki_min, "ki_max": ki_max,
        "kd_min": kd_min, "kd_max": kd_max
    }

    toolbox = init_toolbox(min_max)
    population = toolbox.population(n=pop_size)

    state.update({
        "toolbox": toolbox,
        "population": population,
        "current_idx": 0
    })

    first_ind = population[0]
    return {
        current_params: f"Kp: {first_ind[0]:.4f}, Ki: {first_ind[1]:.4f}, Kd: {first_ind[2]:.4f}",
        fitness_input: "",
        history_output: pd.DataFrame(columns=["Kp", "Ki", "Kd", "Fitness"]),
        best_output: "最佳参数：尚未找到"
    }


def submit_fitness(fitness):
    if not state["running"]:
        return {current_params: "优化未运行"}

    population = state["population"]
    idx = state["current_idx"]
    ind = population[idx]

    # 记录适应度
    ind.fitness.values = (float(fitness),)
    state["history"].append({
        "Kp": ind[0], "Ki": ind[1], "Kd": ind[2], "Fitness": float(fitness)
    })

    # 更新最佳个体
    if state["best_ind"] is None or ind.fitness > state["best_ind"].fitness:
        state["best_ind"] = ind

    # 移动到下一个个体
    state["current_idx"] += 1

    # 检查是否完成当前种群评估
    if state["current_idx"] >= len(population):
        evolve_population()
        state["current_idx"] = 0
        state["current_gen"] += 1

        # 检查终止条件
        if state["current_gen"] >= state["params"]["max_gen"]:
            state["running"] = False
            return {
                current_params: "优化完成！",
                best_output: format_best(state["best_ind"])
            }

    next_ind = population[state["current_idx"]]
    return {
        current_params: f"Kp: {next_ind[0]:.4f}, Ki: {next_ind[1]:.4f}, Kd: {next_ind[2]:.4f}",
        fitness_input: "",
        history_output: pd.DataFrame(state["history"]),
        best_output: format_best(state["best_ind"])
    }


def evolve_population():
    params = state["params"]
    toolbox = state["toolbox"]
    population = state["population"]

    # 选择
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 交叉
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < params["cx_prob"]:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 变异
    for mutant in offspring:
        if random.random() < params["mut_prob"]:
            toolbox.mutate(mutant)
            # 参数裁剪
            mutant[0] = np.clip(mutant[0], *params["kp_range"])
            mutant[1] = np.clip(mutant[1], *params["ki_range"])
            mutant[2] = np.clip(mutant[2], *params["kd_range"])
            del mutant.fitness.values

    state["population"] = offspring


def stop_optimization():
    state["running"] = False
    return {
        current_params: "优化已终止",
        fitness_input: ""
    }


def format_best(ind):
    if ind is None:
        return "最佳参数：尚未找到"
    return f"最佳参数：Kp={ind[0]:.4f}, Ki={ind[1]:.4f}, Kd={ind[2]:.4f} 适应度={ind.fitness.values[0]:.4f}"


with gr.Blocks(title="青云调参", css_paths="./style.css") as demo:
    gr.HTML("<h1 style='text-align: center;'>青云调参</h1><div style='text-align: center;'>适者存千代竞逐，精微处三昧调弦</div>")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                start_btn = gr.Button("开始优化", variant="primary")
                stop_btn = gr.Button("终止优化")

            with gr.Row():
                kp_min = gr.Number(label="Kp最小值", value=0.0)
                kp_max = gr.Number(label="Kp最大值", value=10.0)

            with gr.Row():
                ki_min = gr.Number(label="Ki最小值", value=0.0)
                ki_max = gr.Number(label="Ki最大值", value=10.0)

            with gr.Row():
                kd_min = gr.Number(label="Kd最小值", value=0.0)
                kd_max = gr.Number(label="Kd最大值", value=10.0)

            pop_size = gr.Slider(2, 50, value=5, step=1, label="种群大小")
            max_gen = gr.Slider(1, 50, value=10, step=1, label="迭代次数")
            cx_prob = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="交叉概率")
            mut_prob = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="变异概率")

        with gr.Column(scale=2):
            current_params = gr.Textbox(label="当前参数组合", interactive=False)
            with gr.Row():
                fitness_input = gr.Number(label="输入适应度值")
                submit_btn = gr.Button("提交适应度")

            gr.Markdown("## 最佳参数")
            best_output = gr.Textbox(label="当前最佳参数", interactive=False)

            gr.Markdown("## 优化历史")
            history_output = gr.Dataframe(
                headers=["Kp", "Ki", "Kd", "Fitness"],
                datatype=["number", "number", "number", "number"],
                interactive=False
            )

    start_btn.click(
        start_optimization,
        inputs=[kp_min, kp_max, ki_min, ki_max, kd_min, kd_max,
                pop_size, max_gen, cx_prob, mut_prob],
        outputs=[current_params, fitness_input, history_output, best_output]
    )

    submit_btn.click(
        submit_fitness,
        inputs=fitness_input,
        outputs=[current_params, fitness_input, history_output, best_output]
    )

    stop_btn.click(
        stop_optimization,
        outputs=[current_params, fitness_input]
    )

if __name__ == "__main__":
    demo.launch()
