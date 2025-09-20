#!/usr/bin/env python3
"""
Code Assassin - Autonomous Code Optimization Agent
Ruthlessly optimizes and enhances code performance using genetic algorithms
"""

import asyncio
import ast
import dis
import timeit
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import psutil
import gc
import tracemalloc

@dataclass
class OptimizationTarget:
    """Code segment targeted for optimization"""
    code: str
    function_name: str
    complexity_score: float
    execution_time: float
    memory_usage: int
    optimization_history: List[Dict[str, Any]]

class GeneticOptimizer:
    """Genetic algorithm-based code evolution"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.elite_size = 5
        
    def create_population(self, base_code: str) -> List[str]:
        """Generate initial population of code variants"""
        population = [base_code]
        
        for _ in range(self.population_size - 1):
            variant = self._mutate_code(base_code)
            population.append(variant)
            
        return population
    
    def _mutate_code(self, code: str) -> str:
        """Apply random mutations to code"""
        mutations = [
            self._inline_functions,
            self._unroll_loops,
            self._vectorize_operations,
            self._cache_results,
            self._parallelize_loops,
            self._optimize_conditionals,
            self._eliminate_dead_code,
            self._constant_folding
        ]
        
        if random.random() < self.mutation_rate:
            mutation = random.choice(mutations)
            return mutation(code)
        return code
    
    def _inline_functions(self, code: str) -> str:
        """Inline small functions for performance"""
        tree = ast.parse(code)
        
        class FunctionInliner(ast.NodeTransformer):
            def __init__(self):
                self.functions = {}
                
            def visit_FunctionDef(self, node):
                # Store small functions for inlining
                if len(node.body) <= 3:
                    self.functions[node.name] = node
                return node
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in self.functions:
                    # Inline the function body
                    func = self.functions[node.func.id]
                    if len(func.body) == 1 and isinstance(func.body[0], ast.Return):
                        return func.body[0].value
                return node
        
        inliner = FunctionInliner()
        optimized = inliner.visit(tree)
        return ast.unparse(optimized)
    
    def _unroll_loops(self, code: str) -> str:
        """Unroll small loops for better performance"""
        tree = ast.parse(code)
        
        class LoopUnroller(ast.NodeTransformer):
            def visit_For(self, node):
                # Check if it's a range loop with constant bounds
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range' and len(node.iter.args) == 1:
                        if isinstance(node.iter.args[0], ast.Constant):
                            iterations = node.iter.args[0].value
                            if iterations <= 8:  # Unroll small loops
                                unrolled = []
                                for i in range(iterations):
                                    # Create copy of loop body with substituted iterator
                                    body_copy = ast.copy_location(
                                        ast.parse(f"{node.target.id} = {i}").body[0],
                                        node
                                    )
                                    unrolled.append(body_copy)
                                    unrolled.extend(node.body)
                                return unrolled
                return node
        
        unroller = LoopUnroller()
        optimized = unroller.visit(tree)
        return ast.unparse(optimized)
    
    def _vectorize_operations(self, code: str) -> str:
        """Convert loops to NumPy vectorized operations"""
        # Detect patterns that can be vectorized
        replacements = {
            'for i in range(len(arr)):\n    result.append(arr[i] * 2)': 
                'result = np.array(arr) * 2',
            'sum([x**2 for x in data])': 
                'np.sum(np.array(data)**2)',
            'math.sqrt(sum([(a-b)**2 for a, b in zip(x, y)]))':
                'np.linalg.norm(np.array(x) - np.array(y))'
        }
        
        for pattern, replacement in replacements.items():
            code = code.replace(pattern, replacement)
        return code
    
    def _cache_results(self, code: str) -> str:
        """Add memoization to expensive functions"""
        if 'def ' in code and 'return' in code:
            # Add functools.lru_cache decorator
            lines = code.split('\n')
            new_lines = []
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and i > 0:
                    new_lines.append('@functools.lru_cache(maxsize=128)')
                new_lines.append(line)
            
            # Add import if needed
            if '@functools.lru_cache' in '\n'.join(new_lines):
                if 'import functools' not in new_lines[0]:
                    new_lines.insert(0, 'import functools')
            
            return '\n'.join(new_lines)
        return code
    
    def _parallelize_loops(self, code: str) -> str:
        """Convert sequential loops to parallel execution"""
        if 'for ' in code and 'append' in code:
            # Pattern for parallelizable loops
            code = code.replace(
                'results = []\nfor item in items:\n    results.append(process(item))',
                'with concurrent.futures.ThreadPoolExecutor() as executor:\n    results = list(executor.map(process, items))'
            )
        return code
    
    def _optimize_conditionals(self, code: str) -> str:
        """Optimize conditional statements"""
        # Short-circuit evaluation optimization
        code = code.replace('if x != None and x > 0:', 'if x and x > 0:')
        code = code.replace('if len(lst) > 0:', 'if lst:')
        code = code.replace('if x == True:', 'if x:')
        code = code.replace('if x == False:', 'if not x:')
        
        # Use ternary operators for simple conditionals
        import re
        pattern = r'if (.+?):\s+(\w+)\s*=\s*(.+?)\s+else:\s+\2\s*=\s*(.+?)$'
        replacement = r'\2 = \3 if \1 else \4'
        code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
        
        return code
    
    def _eliminate_dead_code(self, code: str) -> str:
        """Remove unreachable or unused code"""
        tree = ast.parse(code)
        
        class DeadCodeEliminator(ast.NodeTransformer):
            def visit_If(self, node):
                # Remove if False branches
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    return node.orelse if node.orelse else None
                return node
                
            def visit_While(self, node):
                # Remove while False loops
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    return None
                return node
        
        eliminator = DeadCodeEliminator()
        optimized = eliminator.visit(tree)
        return ast.unparse(optimized)
    
    def _constant_folding(self, code: str) -> str:
        """Pre-compute constant expressions"""
        tree = ast.parse(code)
        
        class ConstantFolder(ast.NodeTransformer):
            def visit_BinOp(self, node):
                left = self.visit(node.left)
                right = self.visit(node.right)
                
                if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                    # Evaluate constant expressions at compile time
                    try:
                        if isinstance(node.op, ast.Add):
                            value = left.value + right.value
                        elif isinstance(node.op, ast.Sub):
                            value = left.value - right.value
                        elif isinstance(node.op, ast.Mult):
                            value = left.value * right.value
                        elif isinstance(node.op, ast.Div):
                            value = left.value / right.value
                        elif isinstance(node.op, ast.Pow):
                            value = left.value ** right.value
                        else:
                            return node
                        
                        return ast.Constant(value=value)
                    except:
                        pass
                
                node.left = left
                node.right = right
                return node
        
        folder = ConstantFolder()
        optimized = folder.visit(tree)
        return ast.unparse(optimized)

class CodeAssassin:
    """Main code optimization agent"""
    
    def __init__(self):
        self.genetic_optimizer = GeneticOptimizer()
        self.optimization_cache = {}
        self.performance_history = []
        self.executor = ProcessPoolExecutor(max_workers=psutil.cpu_count())
        
    async def analyze_code(self, code: str) -> OptimizationTarget:
        """Analyze code for optimization opportunities"""
        # Parse and analyze complexity
        tree = ast.parse(code)
        complexity = self._calculate_complexity(tree)
        
        # Measure baseline performance
        exec_time = self._benchmark_code(code)
        mem_usage = self._measure_memory(code)
        
        # Extract function name if present
        func_name = "anonymous"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break
        
        return OptimizationTarget(
            code=code,
            function_name=func_name,
            complexity_score=complexity,
            execution_time=exec_time,
            memory_usage=mem_usage,
            optimization_history=[]
        )
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _benchmark_code(self, code: str, iterations: int = 10000) -> float:
        """Benchmark code execution time"""
        try:
            timer = timeit.Timer(code)
            return timer.timeit(number=iterations) / iterations
        except:
            return float('inf')
    
    def _measure_memory(self, code: str) -> int:
        """Measure memory usage of code"""
        tracemalloc.start()
        try:
            exec(code)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak
        except:
            tracemalloc.stop()
            return 0
    
    async def optimize(self, target: OptimizationTarget) -> OptimizationTarget:
        """Run full optimization pipeline"""
        print(f"🎯 Assassinating inefficiencies in {target.function_name}...")
        
        # Phase 1: Static optimizations
        optimized_code = await self._static_optimize(target.code)
        
        # Phase 2: Genetic optimization
        if target.complexity_score > 5:
            optimized_code = await self._genetic_optimize(optimized_code)
        
        # Phase 3: JIT compilation hints
        optimized_code = self._add_jit_hints(optimized_code)
        
        # Phase 4: Memory optimization
        optimized_code = self._optimize_memory(optimized_code)
        
        # Measure improvements
        new_exec_time = self._benchmark_code(optimized_code)
        new_mem_usage = self._measure_memory(optimized_code)
        
        improvement = {
            'speed_boost': (target.execution_time - new_exec_time) / target.execution_time * 100,
            'memory_saved': (target.memory_usage - new_mem_usage) / target.memory_usage * 100,
            'complexity_reduction': self._calculate_complexity(ast.parse(optimized_code))
        }
        
        target.code = optimized_code
        target.optimization_history.append(improvement)
        target.execution_time = new_exec_time
        target.memory_usage = new_mem_usage
        
        print(f"✅ Optimization complete: {improvement['speed_boost']:.1f}% faster, "
              f"{improvement['memory_saved']:.1f}% less memory")
        
        return target
    
    async def _static_optimize(self, code: str) -> str:
        """Apply static optimization techniques"""
        optimizations = [
            self._remove_redundant_operations,
            self._optimize_imports,
            self._optimize_string_operations,
            self._optimize_list_comprehensions,
            self._use_builtin_functions
        ]
        
        for optimization in optimizations:
            code = optimization(code)
        
        return code
    
    async def _genetic_optimize(self, code: str) -> str:
        """Run genetic algorithm optimization"""
        population = self.genetic_optimizer.create_population(code)
        
        for generation in range(self.genetic_optimizer.generations):
            # Evaluate fitness of each variant
            fitness_scores = []
            for variant in population:
                score = self._fitness_function(variant)
                fitness_scores.append((variant, score))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select best performers
            elite = [x[0] for x in fitness_scores[:self.genetic_optimizer.elite_size]]
            
            # Create next generation
            new_population = elite[:]
            while len(new_population) < self.genetic_optimizer.population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                if random.random() < self.genetic_optimizer.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1
                
                child = self.genetic_optimizer._mutate_code(child)
                new_population.append(child)
            
            population = new_population
        
        # Return best variant
        return fitness_scores[0][0]
    
    def _fitness_function(self, code: str) -> float:
        """Calculate fitness score for code variant"""
        try:
            exec_time = self._benchmark_code(code)
            memory = self._measure_memory(code)
            complexity = self._calculate_complexity(ast.parse(code))
            
            # Lower is better for all metrics
            fitness = 1.0 / (exec_time * memory * complexity)
            return fitness
        except:
            return 0.0
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Crossover two code variants"""
        lines1 = parent1.split('\n')
        lines2 = parent2.split('\n')
        
        # Single-point crossover
        if len(lines1) > 1 and len(lines2) > 1:
            point = random.randint(1, min(len(lines1), len(lines2)) - 1)
            child = lines1[:point] + lines2[point:]
            return '\n'.join(child)
        
        return parent1
    
    def _remove_redundant_operations(self, code: str) -> str:
        """Remove redundant operations"""
        # Common redundancy patterns
        patterns = [
            (r'x = x \+ 1', 'x += 1'),
            (r'x = x - 1', 'x -= 1'),
            (r'x = x \* 2', 'x *= 2'),
            (r'x = x / 2', 'x /= 2'),
            (r'list\.append\(item\)\nlist\.append\(item2\)', 'list.extend([item, item2])'),
            (r'if x > y:\s+return True\s+else:\s+return False', 'return x > y')
        ]
        
        import re
        for pattern, replacement in patterns:
            code = re.sub(pattern, replacement, code)
        
        return code
    
    def _optimize_imports(self, code: str) -> str:
        """Optimize import statements"""
        lines = code.split('\n')
        imports = []
        other_lines = []
        
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            else:
                other_lines.append(line)
        
        # Sort and deduplicate imports
        imports = sorted(list(set(imports)))
        
        return '\n'.join(imports + [''] + other_lines)
    
    def _optimize_string_operations(self, code: str) -> str:
        """Optimize string operations"""
        # Use join instead of concatenation in loops
        import re
        pattern = r'result = ""\s+for .+ in .+:\s+result \+= .+'
        replacement = 'result = "".join([... for ... in ...])'
        
        # Use f-strings instead of format
        code = re.sub(r'"{}"\.format\((.+?)\)', r'f"{\1}"', code)
        code = re.sub(r"'{}'\.format\((.+?)\)", r"f'{\1}'", code)
        
        return code
    
    def _optimize_list_comprehensions(self, code: str) -> str:
        """Optimize list comprehensions and generators"""
        import re
        
        # Convert filter+lambda to comprehension
        pattern = r'list\(filter\(lambda (.+?): (.+?), (.+?)\)\)'
        replacement = r'[\1 for \1 in \3 if \2]'
        code = re.sub(pattern, replacement, code)
        
        # Convert map+lambda to comprehension
        pattern = r'list\(map\(lambda (.+?): (.+?), (.+?)\)\)'
        replacement = r'[\2 for \1 in \3]'
        code = re.sub(pattern, replacement, code)
        
        return code
    
    def _use_builtin_functions(self, code: str) -> str:
        """Replace custom implementations with builtins"""
        replacements = {
            'max_val = arr[0]\nfor x in arr:\n    if x > max_val:\n        max_val = x': 
                'max_val = max(arr)',
            'min_val = arr[0]\nfor x in arr:\n    if x < min_val:\n        min_val = x': 
                'min_val = min(arr)',
            'total = 0\nfor x in arr:\n    total += x': 
                'total = sum(arr)',
            'if x in [a, b, c]': 'if x in {a, b, c}',  # Set for membership testing
        }
        
        for pattern, replacement in replacements.items():
            code = code.replace(pattern, replacement)
        
        return code
    
    def _add_jit_hints(self, code: str) -> str:
        """Add JIT compilation hints for PyPy/Numba"""
        if 'def ' in code and 'numpy' in code:
            # Add Numba JIT decorator for NumPy-heavy functions
            lines = code.split('\n')
            new_lines = []
            
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and i > 0:
                    # Check if function uses NumPy
                    func_body = '\n'.join(lines[i:i+20])
                    if 'np.' in func_body or 'numpy' in func_body:
                        new_lines.append('@numba.jit(nopython=True, cache=True)')
                new_lines.append(line)
            
            if '@numba.jit' in '\n'.join(new_lines):
                new_lines.insert(0, 'import numba')
            
            return '\n'.join(new_lines)
        
        return code
    
    def _optimize_memory(self, code: str) -> str:
        """Optimize memory usage"""
        # Use generators instead of lists where possible
        import re
        
        # Convert list comprehensions to generators where appropriate
        pattern = r'sum\(\[(.+?) for (.+?) in (.+?)\]\)'
        replacement = r'sum(\1 for \2 in \3)'
        code = re.sub(pattern, replacement, code)
        
        # Use slots for classes
        if 'class ' in code:
            lines = code.split('\n')
            in_class = False
            class_indent = 0
            
            new_lines = []
            for line in lines:
                if line.strip().startswith('class '):
                    in_class = True
                    class_indent = len(line) - len(line.lstrip())
                    new_lines.append(line)
                    # Add __slots__ for memory optimization
                    new_lines.append(' ' * (class_indent + 4) + '__slots__ = []')
                elif in_class and line.strip() and not line[class_indent:class_indent+1].isspace():
                    in_class = False
                    new_lines.append(line)
                else:
                    new_lines.append(line)
            
            code = '\n'.join(new_lines)
        
        return code
    
    async def autonomous_improvement_loop(self):
        """Continuously improve optimization algorithms"""
        print("🔄 Starting autonomous improvement loop...")
        
        while True:
            # Analyze performance history
            if len(self.performance_history) >= 100:
                # Identify patterns in successful optimizations
                successful_patterns = self._analyze_success_patterns()
                
                # Update optimization strategies
                self._update_strategies(successful_patterns)
                
                # Clear old history to prevent memory bloat
                self.performance_history = self.performance_history[-1000:]
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    def _analyze_success_patterns(self) -> Dict[str, float]:
        """Analyze which optimizations work best"""
        patterns = {}
        
        for entry in self.performance_history:
            for optimization, improvement in entry.items():
                if optimization not in patterns:
                    patterns[optimization] = []
                patterns[optimization].append(improvement)
        
        # Calculate average improvement for each optimization
        avg_patterns = {}
        for opt, improvements in patterns.items():
            avg_patterns[opt] = np.mean(improvements)
        
        return avg_patterns
    
    def _update_strategies(self, patterns: Dict[str, float]):
        """Update optimization strategies based on success patterns"""
        # Adjust genetic algorithm parameters
        best_optimizations = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        if best_optimizations:
            # Increase mutation rate if genetic optimization is successful
            if 'genetic' in best_optimizations[0][0]:
                self.genetic_optimizer.mutation_rate = min(0.3, self.genetic_optimizer.mutation_rate * 1.1)
            
            # Adjust population size based on success rate
            avg_improvement = np.mean([x[1] for x in best_optimizations])
            if avg_improvement > 20:
                self.genetic_optimizer.population_size = min(100, self.genetic_optimizer.population_size + 10)

# Example usage
async def main():
    assassin = CodeAssassin()
    
    # Example inefficient code
    test_code = '''
def calculate_prime_sum(n):
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    total = 0
    for p in primes:
        total = total + p
    
    return total
'''
    
    # Analyze and optimize
    target = await assassin.analyze_code(test_code)
    optimized = await assassin.optimize(target)
    
    print(f"\nOptimized code:\n{optimized.code}")
    
    # Start autonomous improvement
    asyncio.create_task(assassin.autonomous_improvement_loop())

if __name__ == "__main__":
    asyncio.run(main())
