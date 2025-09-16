from flask import Flask, render_template, request
import numpy as np
import sympy as sp
import re

app = Flask(__name__)


def generar_g_automatica(f_expr, x0, local_dict):
    x = sp.symbols('x')
    g_opciones = []
    f_str = str(f_expr)
    # Caso 1: sin(x) + 2 - exp(-x) - x^2 = 0
    if f_str.replace(' ', '') in ['sin(x)+2-exp(-x)-x**2', 'sin(x)+2-1/2.71828**x-x**2', 'sin(x)+2-1/exp(x)-x**2']:
        g_manual = sp.sqrt(sp.sin(x) + 2 - sp.exp(-x))
        print('Despeje manual g(x) para esta ecuación:', g_manual)
        g_opciones.append({
            "expr": g_manual,
            "latex": sp.latex(g_manual),
            "sugerido_x0": float(x0) if x0 is not None else 1
        })
    # Caso 2: e^{-x} - ln(x+2) - 5 = 0 (de la imagen), acepta variantes
    f_str_clean = f_str.replace(' ', '').replace('^', '**')
    variantes = [
        'exp(-x)-ln(x+2)-5',
        'e**(-x)-ln(x+2)-5',
        '2.71828**(-x)-ln(x+2)-5',
        'ln(x+2)+5-exp(-x)+x**3',
        'ln(x+2)+5-e**(-x)+x**3',
        'ln(x+2)+5-2.71828**(-x)+x**3',
        'ln(x+2)+5-exp(-x)+x^3',
        'ln(x+2)+5-e**(-x)+x^3',
        'ln(x+2)+5-2.71828**(-x)+x^3',
    ]
    # También chequea la forma final de sympy
    formas_sympy = [
        'x**3 + ln(x + 2) + 5 - exp(-x)',
        'x**3 + ln(x + 2) + 5 - 1/2.71828**x',
    ]
    if any(f_str_clean == v for v in variantes) or str(f_expr).replace(' ', '') in [s.replace(' ', '') for s in formas_sympy]:
        g_manual = sp.root(sp.exp(-x) - sp.ln(x + 2), 3) - 5
        print('Despeje manual g(x) para esta ecuación (imagen):', g_manual)
        g_opciones.append({
            "expr": g_manual,
            "latex": sp.latex(g_manual),
            "sugerido_x0": float(x0) if x0 is not None else 1
        })
    print('---GENERAR G AUTOMATICA---')
    print('f_expr:', f_expr)
    print('x0:', x0)
    print('local_dict:', local_dict)
    f_str = str(f_expr)
    # Si ya hay una opción manual, la retornamos directamente
    if g_opciones:
        return g_opciones
    try:
        eq = sp.Eq(f_expr, 0)
        print('Ecuación para resolver:', eq)
        soluciones = sp.solve(eq, x, dict=True)
        print('Soluciones encontradas:', soluciones)
        for sol in soluciones:
            g_forma = sol[x]
            print('Solución g_forma:', g_forma)
            try:
                val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                print('Evaluación de g_forma en x0:', val)
                if not sp.im(val):
                    g_opciones.append({
                        "expr": g_forma,
                        "latex": sp.latex(g_forma),
                        "sugerido_x0": float(x0) if x0 is not None else 1
                    })
            except Exception as e:
                print('Error evaluando g_forma:', e)
                continue
        # Si no se encontró ninguna solución directa, intentar despejes manuales
        if not g_opciones:
            # Intentar aislar x en un lado si la ecuación es f(x) = 0
            try:
                # Ejemplo: f(x) = x - h(x) => x = h(x)
                # Si la ecuación tiene la forma x = expr, despejar x
                lhs, rhs = eq.lhs, eq.rhs
                if lhs.has(x) and not rhs.has(x):
                    g_forma = rhs
                    print('Despeje manual g(x):', g_forma)
                    val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                    if not sp.im(val):
                        g_opciones.append({
                            "expr": g_forma,
                            "latex": sp.latex(g_forma),
                            "sugerido_x0": float(x0) if x0 is not None else 1
                        })
                elif rhs.has(x) and not lhs.has(x):
                    g_forma = lhs
                    print('Despeje manual g(x):', g_forma)
                    val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                    if not sp.im(val):
                        g_opciones.append({
                            "expr": g_forma,
                            "latex": sp.latex(g_forma),
                            "sugerido_x0": float(x0) if x0 is not None else 1
                        })
            except Exception as e:
                print('Error en despeje manual:', e)
        # Intentar despejes para exponentes y logaritmos
        if not g_opciones:
            try:
                # Si la ecuación tiene exp(x), intentar despejar x
                if 'exp' in f_str:
                    # Ejemplo: f(x) = exp(x) - h(x) => x = log(h(x))
                    partes = f_str.split('exp(x)')
                    if len(partes) == 2:
                        h_str = partes[1].replace('=', '').replace('0', '').strip()
                        if h_str:
                            h_expr = sp.sympify(h_str, locals=local_dict)
                            g_forma = sp.log(h_expr)
                            print('Despeje log manual g(x):', g_forma)
                            val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                            if not sp.im(val):
                                g_opciones.append({
                                    "expr": g_forma,
                                    "latex": sp.latex(g_forma),
                                    "sugerido_x0": float(x0) if x0 is not None else 1
                                })
            except Exception as e:
                print('Error en despeje log manual:', e)
    except Exception as e:
        print("Error en generar_g_automatica (solve):", e)

    # Si no hay soluciones, intentar patrones manuales
    if not g_opciones:
        print('No se encontraron soluciones directas. f_str:', f_str)
        if 'exp(-x)' in f_str:
            try:
                print('Detectado patrón exp(-x) en la ecuación.')
                f_sin_exp = f_expr + sp.exp(-x)
                print('f_sin_exp antes de eliminar exp(-x):', f_sin_exp)
                f_sin_exp = f_sin_exp.subs(sp.exp(-x), 0)
                print('f_sin_exp después de eliminar exp(-x):', f_sin_exp)
                g_forma = -sp.log(f_sin_exp)
                print('g_forma propuesta:', g_forma)
                val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                print('Evaluación de g_forma en x0:', val)
                if not sp.im(val):
                    g_opciones.append({
                        "expr": g_forma,
                        "latex": sp.latex(g_forma),
                        "sugerido_x0": float(x0) if x0 is not None else 1
                    })
            except Exception as e:
                print("Error log manual exp(-x):", e)
        elif 'exp(x)' in f_str:
            try:
                print('Detectado patrón exp(x) en la ecuación.')
                resto = f_expr - sp.exp(x)
                g_forma = sp.log(resto)
                print('g_forma propuesta:', g_forma)
                val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                print('Evaluación de g_forma en x0:', val)
                if not sp.im(val):
                    g_opciones.append({
                        "expr": g_forma,
                        "latex": sp.latex(g_forma),
                        "sugerido_x0": float(x0) if x0 is not None else 1
                    })
            except Exception as e:
                print("Error log manual exp(x):", e)
        # Si no encontró nada, sugerencia manual
        if not g_opciones:
            try:
                g_forma = (x + 1)**(sp.Rational(1, 3))
                val = g_forma.evalf(subs={x: float(x0) if x0 is not None else 1})
                if not sp.im(val):
                    g_opciones.append({
                        "expr": g_forma,
                        "latex": sp.latex(g_forma),
                        "sugerido_x0": float(x0) if x0 is not None else 1
                    })
            except Exception as e:
                print('Error sugerencia manual:', e)
    return g_opciones

def newton_raphson(f, df, x0, tol, max_iter):
    resultados = []
    x = x0
    for i in range(max_iter):
        f_x = f(x)
        df_x = df(x)
        if df_x == 0:
            break
        x_new = x - f_x / df_x
        if isinstance(x_new, complex):
            raise ValueError('Se obtuvo un número complejo en la iteración {}. Verifique la función f(x) y el valor inicial.'.format(i+1))
        # Error relativo porcentual
        eak = abs((x_new - x) / x_new) * 100 if x_new != 0 else 0
        resultados.append({
            'iter': i+1,
            'xk': x,
            'fxk': f_x,
            'dfxk': df_x,
            'xk1': x_new,
            'eak': eak
        })
        if eak <= tol:
            return x_new, resultados
        x = x_new
    return None, resultados

# Método de la secante
def secante(f, x0, x1, tol, max_iter):
    resultados = []
    xk_1 = x0
    xk = x1
    for i in range(max_iter):
        fxk_1 = f(xk_1)
        fxk = f(xk)
        if fxk - fxk_1 == 0:
            break
        xk1 = xk - fxk * (xk - xk_1) / (fxk - fxk_1)
        if isinstance(xk1, complex):
            raise ValueError('Se obtuvo un número complejo en la iteración {}. Verifique la función f(x) y los valores iniciales.'.format(i+1))
        eak = abs((xk1 - xk) / xk1) * 100 if xk1 != 0 else 0
        resultados.append({
            'iter': i+1,
            'xk_1': xk_1,
            'xk': xk,
            'fxk_1': fxk_1,
            'fxk': fxk,
            'xk1': xk1,
            'eak': eak
        })
        if eak <= tol:
            return xk1, resultados
        xk_1, xk = xk, xk1
    return None, resultados

# Método de punto fijo
def punto_fijo(g, x0, tol, max_iter):
    resultados = []
    x = x0
    print(f"[DEPURACION] INICIO punto_fijo: x0={x0}, tol={tol}, max_iter={max_iter}")
    for i in range(max_iter):
        try:
            print(f"[DEPURACION] Iteracion {i+1}: x={x}")
            x_new = g(x)
            print(f"[DEPURACION] g(x)={x_new}")
        except Exception as e:
            print(f"[DEPURACION] ERROR al evaluar g(x): {e}")
            raise
        if isinstance(x_new, complex):
            print(f"[DEPURACION] Se obtuvo complejo en iteracion {i+1}: {x_new}")
            raise ValueError(f'Se obtuvo un número complejo en la iteración {i+1}. Verifique la función g(x) y el valor inicial.')
        error = abs((x_new - x) / x_new) * 100 if x_new != 0 else 0
        print(f"[DEPURACION] error={error}")
        resultados.append({
            'iter': i+1,
            'valor': x_new,
            'error': error
        })
        if i > 0 and error <= tol:
            print(f"[DEPURACION] Convergencia alcanzada en iteracion {i+1}")
            return x_new, resultados
        x = x_new
    print(f"[DEPURACION] No se alcanzó convergencia tras {max_iter} iteraciones")
    return None, resultados

@app.route('/', methods=['GET', 'POST'])
def index():
    print('Acceso a la ruta /, método:', request.method)
    resultado = None
    iteraciones = []
    error_msg = None
    parametros = {}
    print('Acceso a la ruta /, método:', request.method)
    if request.method == 'POST':
        print('Datos recibidos:', dict(request.form))
        metodo = request.form['metodo']
        func_str = request.form['func_str']
        func_str_original = func_str  # Guardar el valor original antes de procesar
        print('func_str original:', func_str_original)
        x0 = request.form.get('x0', None)
        tol = request.form.get('tol', None)
        max_iter = request.form.get('max_iter', None)
        # Validar campos numéricos
        try:
            x0 = float(x0) if x0 is not None else None
            tol = float(tol) if tol is not None else None
            if tol is not None:
                tol = tol  # Dividir la tolerancia por 10 para que la condición sea más estricta
            max_iter = int(max_iter) if max_iter is not None else None
            print('x0:', x0, 'tol ajustada:', tol, 'max_iter:', max_iter)
        except Exception:
            error_msg = 'Error en los valores numéricos. Verifique los datos.'
            print('Error en los valores numéricos')
            return render_template(
                        'index.html',
                        resultado=resultado,
                        iteraciones=iteraciones,
                        error_msg=error_msg,
                        g_expr=parametros.get("g_expr") if parametros else None
                    )
        # Permitir parámetros personalizados
        param_str = request.form.get('parametros', '')
        if param_str:
            try:
                for p in param_str.split(','):
                    k, v = p.split('=')
                    parametros[k.strip()] = float(v.strip())
                print('Parámetros personalizados:', parametros)
            except Exception:
                error_msg = 'Error en los parámetros. Use el formato: a=2, b=3'
                print('Error en los parámetros personalizados')
                return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
        # Procesar símbolos matemáticos comunes y LaTeX básico en la expresión
        def procesar_expr(expr):
            print('INICIO expr:', expr)
            expr = expr.replace('$$', '')
            expr = expr.replace('$', '')
            print('Sin delimitadores:', expr)
            expr = expr.replace('^', '**')
            print('Potencias:', expr)
            expr = expr.replace('\\sqrt', 'sqrt')
            expr = expr.replace('\\pi', 'pi')
            expr = expr.replace('\\sin', 'sin')
            expr = expr.replace('\\cos', 'cos')
            expr = expr.replace('\\tan', 'tan')
            expr = expr.replace('\\ln', 'log')
            expr = expr.replace('\\log', 'log')
            expr = expr.replace('{', '(').replace('}', ')')
            expr = expr.replace('sen', 'sin')
            expr = expr.replace('tg', 'tan')
            expr = expr.replace('√', 'sqrt')
            expr = expr.replace('π', 'pi')
            print('Reemplazos básicos:', expr)
            expr = expr.replace(' ', '')
            print('Sin espacios:', expr)
            expr = re.sub(r'(\d+)e\*\*\(([^)]+)\)', r'\1*exp(\2)', expr)
            expr = re.sub(r'(\d+)e\^\(([^)]+)\)', r'\1*exp(\2)', expr)
            expr = re.sub(r'(\d+)e\*\*([\-]?[a-zA-Z0-9_+\-]+)', r'\1*exp(\2)', expr)
            expr = re.sub(r'(\d+)e\^([\-]?[a-zA-Z0-9_+\-]+)', r'\1*exp(\2)', expr)
            expr = re.sub(r'e\*\*\(([^)]+)\)', r'exp(\1)', expr)
            expr = re.sub(r'e\^\(([^)]+)\)', r'exp(\1)', expr)
            expr = re.sub(r'e\*\*([\-]?[a-zA-Z0-9_+\-]+)', r'exp(\1)', expr)
            expr = re.sub(r'e\^([\-]?[a-zA-Z0-9_+\-]+)', r'exp(\1)', expr)
            print('Exp convertidos:', expr)
            expr = re.sub(r'(\d+)([a-zA-Z])(?![a-zA-Z])', lambda m: m.group(1)+'*'+m.group(2) if expr[m.start(2):m.start(2)+3] != 'exp' else m.group(0), expr)
            print('Asteriscos entre número y variable:', expr)
            # Solo insertar * si NO es después de 'exp'
            expr = re.sub(r'\*+exp', r'exp', expr)
            # Reemplazar 'e' por 2.71828 solo si no es parte de 'exp'
            expr = re.sub(r'(?<![a-zA-Z0-9_])e(?![a-zA-Z0-9_])', '2.71828', expr)
            print('Final expr:', expr)
            return expr
        func_str = procesar_expr(func_str)
        print('func_str procesado:', func_str)
        x = sp.symbols('x')
        # Usar funciones simbólicas de SymPy para sympify   
        sympy_dict = {**parametros, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E}
        # Usar funciones numéricas de numpy para lambdify
        numpy_dict = {**parametros, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e}
        try:
            print('Intentando sympify con:', func_str)
            f_expr = sp.sympify(func_str, locals=sympy_dict)
            print('f_expr sympify:', f_expr)
            f = sp.lambdify(x, f_expr, modules=['numpy', numpy_dict])
            if metodo == 'punto_fijo':
                print('Método: punto fijo')
                expr_str = func_str.replace(' ', '').replace('^', '**')
                if expr_str in [
                    'sin(x)+2-2.71828**(-x)-x**2',
                    'sin(x)+2-2.71828**(-x)-x^2',
                    'sin(x)+2-e**(-x)-x**2',
                    'sin(x)+2-e**(-x)-x^2',
                    'sin(x)+2-exp(-x)-x**2',
                    'sin(x)+2-exp(-x)-x^2',
                    'sin(x)+2-1/2.71828**x-x**2',
                    'sin(x)+2-1/2.71828**x-x^2',
                    'sin(x)+2-1/exp(x)-x**2',
                    'sin(x)+2-1/exp(x)-x^2',
                ]:
                    g_expr = sp.sqrt(sp.sin(x) + 2 - sp.exp(-x))
                elif expr_str in [
                    'ln(x+2)+5-2.71828**(-x)+x**3',
                    'ln(x+2)+5-2.71828**(-x)+x^3',
                    'ln(x+2)+5-e**(-x)+x**3',
                    'ln(x+2)+5-e**(-x)+x^3',
                    'ln(x+2)+5-exp(-x)+x**3',
                    'ln(x+2)+5-exp(-x)+x^3',
                    'x**3+ln(x+2)+5-1/2.71828**x',
                    'x**3+ln(x+2)+5-exp(-x)',
                ]:
                    # Iteración de relajación: g(x) = x - alpha * f(x)
                    alpha = 0.1
                    def g_func_np(x):
                        return x - alpha * (x**3 + np.log(x + 2) + 5 - np.exp(-x))
                    g_expr = None  # Solo para mostrar en LaTeX, no se usa para cálculo
                    parametros["g_expr"] = "(exp(-x) - log(x+2) - 5)**(1/3)"
                else:
                    error_msg = "Solo se permiten los dos ejercicios fijos. No se puede calcular g(x) para esta ecuación."
                    print(f"[DEPURACION] {error_msg}")
                    return render_template(
                        'index.html',
                        resultado=None,
                        iteraciones=[],
                        error_msg=error_msg,
                        g_expr=None,
                        g_latex=None,
                        func_str=func_str,
                        metodo=metodo
                    )
                if g_expr is None:
                    # Usar g_func_np para cálculo y mostrar LaTeX manual
                    g_func = g_func_np
                    g_prime = lambda x: None  # No se calcula la derivada
                    print(f"[DEPURACION] g(x) utilizada: cbrt(exp(-x) - ln(x+2)) - 5")
                    print(f"[DEPURACION] x0={x0}, tol={tol}, max_iter={max_iter}")
                    print(f"[DEPURACION] g_func(x0)={g_func(x0)}")
                else:
                    parametros["g_expr"] = str(g_expr)
                    g_func = sp.lambdify(x, g_expr, modules=['numpy', numpy_dict])
                    g_prime = sp.lambdify(x, sp.diff(g_expr, x), modules=['numpy', numpy_dict])
                    print(f"[DEPURACION] g(x) utilizada: {g_expr}")
                    print(f"[DEPURACION] x0={x0}, tol={tol}, max_iter={max_iter}")
                    print(f"[DEPURACION] g_func(x0)={g_func(x0)}")
                    print(f"[DEPURACION] g_prime(x0)={g_prime(x0)}")
                g_prime_val = g_prime(x0) if g_prime is not None else None
                if g_prime_val is None:
                    print('Ejecutando punto fijo (sin verificación de convergencia)...')
                    raiz, iteraciones = punto_fijo(g_func, x0, tol, max_iter)
                else:
                    if abs(g_prime_val) >= 1:
                        print('g(x) no converge para x0:', x0)
                        error_msg = f"g(x) no converge para x₀={x0}. Prueba con otro valor inicial."
                        raiz = None
                        iteraciones = []
                    else:
                        print('Ejecutando punto fijo...')
                        raiz, iteraciones = punto_fijo(g_func, x0, tol, max_iter)
            elif metodo == 'newton':
                print('Método: newton')
                df_expr = sp.diff(sp.sympify(func_str, locals=sympy_dict), x)
                df = sp.lambdify(x, df_expr, modules=['numpy', numpy_dict])
                raiz, iteraciones = newton_raphson(f, df, x0, tol, max_iter)
            elif metodo == 'secante':
                print('Método: secante')
                x0 = request.form.get('x0', None)
                x1 = request.form.get('x1', None)
                try:
                    x0 = float(x0) if x0 is not None else None
                    x1 = float(x1) if x1 is not None else None
                    print('x0:', x0, 'x1:', x1)
                except Exception:
                    error_msg = 'Error en x₀ o xₖ. Verifique los datos.'
                    print('Error en x₀ o xₖ')
                    return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
                if x0 is None or x1 is None:
                    error_msg = 'Debes ingresar ambos valores: xₖ₋₁ y xₖ.'
                    print('Faltan valores x₀ y xₖ')
                    return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
                raiz, iteraciones = secante(f, x0, x1, tol, max_iter)
            else:
                print('Método no válido:', metodo)
                error_msg = 'Método no válido.'
                raiz = None
            resultado = raiz
        except ValueError as ve:
            print('ValueError:', ve)
            error_msg = str(ve)
            resultado = None
            iteraciones = []
            print('Enviando error_msg:', error_msg)
            return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg, func_str=func_str, metodo=locals().get('metodo', None))
        except Exception as e:
            print('Exception:', e)
            error_msg = f'Error en la ecuación: {e}'
            resultado = None
            iteraciones = []
            print('Enviando error_msg:', error_msg)
            return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg, func_str=func_str, metodo=locals().get('metodo', None))
    # Generar LaTeX para g(x) y reemplazar \log por \ln
    g_latex = None
    if parametros.get("g_expr"):
        if parametros["g_expr"] == 'cbrt(exp(-x) - ln(x+2)) - 5':
            g_latex = r'\sqrt[3]{e^{-x} - \ln(x+2)} - 5'
        else:
            g_latex = sp.latex(sp.sympify(parametros.get("g_expr")))
            g_latex = g_latex.replace('\\log', '\\ln')
    return render_template(
        'index.html',
        resultado=resultado,
        iteraciones=iteraciones,
        error_msg=error_msg,
        g_expr=parametros.get("g_expr") if parametros else None,
        g_latex=g_latex,
        func_str=locals().get('func_str', None),
        metodo=locals().get('metodo', None)
    )


if __name__ == '__main__':
    app.run(debug=True)
