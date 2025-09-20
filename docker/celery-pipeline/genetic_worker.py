#!/usr/bin/env python3
"""
Genetic Algorithm Worker for ORACLE1
Prompt optimization and evolutionary computation
"""

import json
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog
from celery import Task
from celery_app import app
from deap import algorithms, base, creator, tools
from pydantic import BaseModel
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import redis

# Configure structured logging
logger = structlog.get_logger("genetic_worker")

class GeneticTask(BaseModel):
    """Genetic algorithm task model"""
    task_id: str
    optimization_type: str
    parameters: Dict[str, Any]
    fitness_function: str
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: str = "tournament"
    target_fitness: Optional[float] = None
    constraints: Dict[str, Any] = {}

class PromptGenome(BaseModel):
    """Prompt genome for evolutionary optimization"""
    template: str
    parameters: Dict[str, Any]
    components: List[str]
    fitness_score: float = 0.0
    generation: int = 0
    mutations: int = 0

class OptimizationResult(BaseModel):
    """Optimization result model"""
    task_id: str
    best_individual: Dict[str, Any]
    best_fitness: float
    generation_reached: int
    total_evaluations: int
    convergence_history: List[float]
    execution_time: float
    metadata: Dict[str, Any]

class GeneticOptimizer:
    """Genetic algorithm optimization engine"""

    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=2)
        self.toolbox = base.Toolbox()
        self.setup_operators()

    def setup_operators(self):
        """Setup DEAP genetic operators"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def optimize_prompt_template(self, task: GeneticTask) -> OptimizationResult:
        """Optimize prompt template using genetic algorithms"""
        try:
            logger.info("Starting prompt optimization", task_id=task.task_id)
            start_time = time.time()

            # Extract parameters
            base_template = task.parameters.get('base_template', '')
            variable_components = task.parameters.get('variable_components', [])
            evaluation_criteria = task.parameters.get('evaluation_criteria', {})

            # Setup genetic algorithm
            self._setup_prompt_optimization(base_template, variable_components)

            # Create initial population
            population = self.toolbox.population(n=task.population_size)

            # Track convergence
            convergence_history = []
            best_fitness = float('-inf')

            # Evolution loop
            for generation in range(task.generations):
                logger.info(f"Processing generation {generation + 1}/{task.generations}")

                # Evaluate fitness
                fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit

                # Track best fitness
                current_best = max(ind.fitness.values[0] for ind in population)
                convergence_history.append(current_best)

                if current_best > best_fitness:
                    best_fitness = current_best
                    logger.info(f"New best fitness: {best_fitness}")

                # Check convergence
                if task.target_fitness and best_fitness >= task.target_fitness:
                    logger.info("Target fitness reached", target=task.target_fitness)
                    break

                # Selection and reproduction
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))

                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < task.crossover_rate:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if random.random() < task.mutation_rate:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Replace population
                population[:] = offspring

            # Get best individual
            best_individual = tools.selBest(population, 1)[0]

            execution_time = time.time() - start_time

            result = OptimizationResult(
                task_id=task.task_id,
                best_individual=self._decode_individual(best_individual, base_template, variable_components),
                best_fitness=best_individual.fitness.values[0],
                generation_reached=generation + 1,
                total_evaluations=len(population) * (generation + 1),
                convergence_history=convergence_history,
                execution_time=execution_time,
                metadata={
                    "optimization_type": "prompt_template",
                    "population_size": task.population_size,
                    "final_generation": generation + 1
                }
            )

            # Cache result
            self._cache_optimization_result(result)

            return result

        except Exception as e:
            logger.error("Prompt optimization failed", task_id=task.task_id, error=str(e))
            raise

    def _setup_prompt_optimization(self, base_template: str, variable_components: List[str]):
        """Setup genetic operators for prompt optimization"""

        def create_individual():
            """Create a random individual (prompt configuration)"""
            individual = []
            for component in variable_components:
                if component['type'] == 'text_choice':
                    choice = random.choice(component['options'])
                    individual.append(choice)
                elif component['type'] == 'parameter':
                    value = random.uniform(component['min'], component['max'])
                    individual.append(value)
                elif component['type'] == 'boolean':
                    individual.append(random.choice([True, False]))
            return creator.Individual(individual)

        def evaluate_prompt(individual):
            """Evaluate prompt fitness"""
            try:
                # Decode individual to prompt
                prompt = self._decode_individual(individual, base_template, variable_components)

                # Simple fitness evaluation (can be replaced with actual LLM evaluation)
                fitness = self._calculate_prompt_fitness(prompt)

                return (fitness,)

            except Exception as e:
                logger.warning("Fitness evaluation failed", error=str(e))
                return (0.0,)

        # Register functions
        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate_prompt)

    def _decode_individual(self, individual: List, base_template: str, variable_components: List[str]) -> Dict[str, Any]:
        """Decode genetic individual to prompt configuration"""
        prompt_config = {
            "template": base_template,
            "components": {},
            "parameters": {}
        }

        for i, component in enumerate(variable_components):
            value = individual[i]
            prompt_config["components"][component['name']] = value

            if component['type'] == 'parameter':
                prompt_config["parameters"][component['name']] = value

        return prompt_config

    def _calculate_prompt_fitness(self, prompt_config: Dict[str, Any]) -> float:
        """Calculate fitness score for prompt configuration"""
        try:
            # Simple heuristic fitness function
            fitness = 0.0

            # Length penalty/bonus
            template_length = len(prompt_config["template"])
            if 50 <= template_length <= 500:
                fitness += 0.2
            else:
                fitness -= 0.1

            # Component diversity bonus
            component_count = len(prompt_config["components"])
            fitness += min(component_count * 0.1, 0.3)

            # Parameter optimization bonus
            parameter_count = len(prompt_config["parameters"])
            fitness += min(parameter_count * 0.05, 0.2)

            # Add some randomness to simulate real evaluation
            fitness += random.uniform(-0.1, 0.1)

            return max(0.0, fitness)

        except Exception as e:
            logger.error("Fitness calculation failed", error=str(e))
            return 0.0

    def optimize_hyperparameters(self, task: GeneticTask) -> OptimizationResult:
        """Optimize hyperparameters using genetic algorithms or Optuna"""
        try:
            logger.info("Starting hyperparameter optimization", task_id=task.task_id)
            start_time = time.time()

            # Use Optuna for advanced hyperparameter optimization
            study = optuna.create_study(direction='maximize')

            def objective(trial):
                """Optuna objective function"""
                params = {}

                # Define parameter space
                param_space = task.parameters.get('parameter_space', {})
                for param_name, param_config in param_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

                # Evaluate parameters
                return self._evaluate_hyperparameters(params, task.parameters)

            # Run optimization
            study.optimize(objective, n_trials=task.parameters.get('n_trials', 100))

            execution_time = time.time() - start_time

            result = OptimizationResult(
                task_id=task.task_id,
                best_individual=study.best_params,
                best_fitness=study.best_value,
                generation_reached=len(study.trials),
                total_evaluations=len(study.trials),
                convergence_history=[trial.value for trial in study.trials if trial.value is not None],
                execution_time=execution_time,
                metadata={
                    "optimization_type": "hyperparameters",
                    "study_name": study.study_name,
                    "n_trials": len(study.trials)
                }
            )

            # Cache result
            self._cache_optimization_result(result)

            return result

        except Exception as e:
            logger.error("Hyperparameter optimization failed", task_id=task.task_id, error=str(e))
            raise

    def _evaluate_hyperparameters(self, params: Dict[str, Any], task_params: Dict[str, Any]) -> float:
        """Evaluate hyperparameter configuration"""
        try:
            # Simple example using RandomForest (replace with actual model)
            from sklearn.datasets import make_classification
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier

            # Generate sample data
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

            # Create model with hyperparameters
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=42
            )

            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()

        except Exception as e:
            logger.error("Hyperparameter evaluation failed", error=str(e))
            return 0.0

    def _cache_optimization_result(self, result: OptimizationResult):
        """Cache optimization result in Redis"""
        try:
            self.redis_client.setex(
                f"optimization_result:{result.task_id}",
                3600,  # 1 hour TTL
                result.json()
            )
            logger.info("Optimization result cached", task_id=result.task_id)

        except Exception as e:
            logger.warning("Failed to cache result", task_id=result.task_id, error=str(e))

    def get_cached_result(self, task_id: str) -> Optional[OptimizationResult]:
        """Get cached optimization result"""
        try:
            cached_data = self.redis_client.get(f"optimization_result:{task_id}")
            if cached_data:
                return OptimizationResult.parse_raw(cached_data)
            return None

        except Exception as e:
            logger.warning("Failed to get cached result", task_id=task_id, error=str(e))
            return None

# Initialize genetic optimizer
genetic_optimizer = GeneticOptimizer()

class GeneticOptimizationTask(Task):
    """Custom Celery task for genetic optimization"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Genetic optimization task failed",
                    task_id=task_id,
                    exception=str(exc),
                    traceback=str(einfo))

@app.task(bind=True, base=GeneticOptimizationTask, queue='genetic_optimization')
def optimize_prompts(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize prompts using genetic algorithms"""
    try:
        logger.info("Processing prompt optimization task", task_id=self.request.id)

        task = GeneticTask(**task_data)

        # Check cache first
        cached_result = genetic_optimizer.get_cached_result(task.task_id)
        if cached_result:
            logger.info("Returning cached result", task_id=task.task_id)
            return cached_result.dict()

        # Run optimization
        result = genetic_optimizer.optimize_prompt_template(task)
        return result.dict()

    except Exception as e:
        logger.error("Prompt optimization failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=GeneticOptimizationTask, queue='genetic_optimization')
def optimize_hyperparameters(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize hyperparameters using genetic algorithms and Optuna"""
    try:
        logger.info("Processing hyperparameter optimization task", task_id=self.request.id)

        task = GeneticTask(**task_data)

        # Check cache first
        cached_result = genetic_optimizer.get_cached_result(task.task_id)
        if cached_result:
            logger.info("Returning cached result", task_id=task.task_id)
            return cached_result.dict()

        # Run optimization
        result = genetic_optimizer.optimize_hyperparameters(task)
        return result.dict()

    except Exception as e:
        logger.error("Hyperparameter optimization failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=GeneticOptimizationTask, queue='genetic_optimization')
def evolve_solution(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evolve general solutions using genetic algorithms"""
    try:
        logger.info("Processing solution evolution task", task_id=self.request.id)

        task = GeneticTask(**task_data)

        # Custom evolution logic based on problem type
        problem_type = task.parameters.get('problem_type', 'optimization')

        if problem_type == 'prompt_optimization':
            result = genetic_optimizer.optimize_prompt_template(task)
        elif problem_type == 'parameter_tuning':
            result = genetic_optimizer.optimize_hyperparameters(task)
        else:
            # Generic optimization
            result = genetic_optimizer.optimize_prompt_template(task)

        return result.dict()

    except Exception as e:
        logger.error("Solution evolution failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=GeneticOptimizationTask, queue='genetic_optimization')
def benchmark_optimization(self, benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark different optimization approaches"""
    try:
        logger.info("Processing optimization benchmark", task_id=self.request.id)

        results = {}
        approaches = benchmark_config.get('approaches', ['genetic', 'optuna'])

        for approach in approaches:
            start_time = time.time()

            if approach == 'genetic':
                task = GeneticTask(**benchmark_config['genetic_config'])
                result = genetic_optimizer.optimize_prompt_template(task)
            elif approach == 'optuna':
                task = GeneticTask(**benchmark_config['optuna_config'])
                result = genetic_optimizer.optimize_hyperparameters(task)

            execution_time = time.time() - start_time

            results[approach] = {
                "best_fitness": result.best_fitness,
                "execution_time": execution_time,
                "generations": result.generation_reached,
                "evaluations": result.total_evaluations
            }

        # Find best approach
        best_approach = max(results.keys(), key=lambda k: results[k]['best_fitness'])

        return {
            "benchmark_id": self.request.id,
            "results": results,
            "best_approach": best_approach,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Optimization benchmark failed", task_id=self.request.id, error=str(e))
        raise

if __name__ == "__main__":
    # Test genetic optimization functionality
    test_task = GeneticTask(
        task_id="test_genetic_001",
        optimization_type="prompt_optimization",
        parameters={
            "base_template": "Generate a {style} response about {topic}",
            "variable_components": [
                {"name": "style", "type": "text_choice", "options": ["formal", "casual", "technical"]},
                {"name": "topic", "type": "text_choice", "options": ["AI", "science", "technology"]}
            ]
        },
        population_size=20,
        generations=10
    )

    print("Genetic optimization worker ready for evolutionary computation")