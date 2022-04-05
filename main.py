from ensurepip import bootstrap
from filecmp import DEFAULT_IGNORES
from flask import  request, make_response, redirect, render_template, session, url_for, flash
from flask_login import login_required, current_user
from app import create_app
from app.forms import LoginForm, TodoForm, DeleteTodoForm
from app.firestore_service import *
from sdrscanfm.fm import *
from sdrscantv.tv import *

app = create_app()



@app.cli.command()
def test():
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner().run(tests)




@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', error = error)



@app.route('/')
def index():
    user_ip = request.remote_addr
    response = make_response(redirect('/hello'))
    session['user_ip'] = user_ip
    return response



@app.route('/hello', methods=['GET', 'POST'])
@login_required
def hello():
    user_ip = session.get('user_ip')
    username = current_user.id
    todo_form = TodoForm()
    delete_form = DeleteTodoForm()
    

    context = {
        'user_ip': user_ip,
        'todos': get_todos(user_id=username),
        'username': username,
        'todo_form': todo_form,
        'delete_form': delete_form,
    }

    if todo_form.validate_on_submit():
        put_todo(user_id=username, description=todo_form.description.data)

        flash('Tu tarea se creo con éxito!')

        return redirect(url_for('hello'))

    return render_template('hello.html', **context)

@app.route('/fm', methods=['GET', 'POST'])
@login_required
def fm():
    user_ip = session.get('user_ip')
    username = current_user.id

    data1_fm, espuria, descision =primera_iteracion_fm()
    '''data2=segunda_iteracion_fm()
    data3=tercera_iteracion_fm()
    data4=cuarta_iteracion_fm()
    data5=quinta_iteracion_fm()'''
 

    context = {
        'username': username,
        'data1_fm': data1_fm,

    }
    
    put_signal_fm(user_id=username, signal=data1_fm)
    put_alarma_fm(user_id=username, parasita=espuria)
    print(espuria)
    print(descision)
    return render_template('fm.html',**context)

@app.route('/tv_uhf', methods=['GET', 'POST'])
@login_required
def tv_uhf():
    user_ip = session.get('user_ip')
    username = current_user.id

    data1_tv, espuria, descision =primera_iteracion_tv()
    '''data2=segunda_iteracion_fm()
    data3=tercera_iteracion_fm()
    data4=cuarta_iteracion_fm()
    data5=quinta_iteracion_fm()'''


    context = {
        'username': username,
        'data1_fm': data1_tv,

    }
    
    #put_signal_tv(user_id=username, signal=data1_tv)
    #put_alarma_tv(user_id=username, parasita=espuria)
    print(data1_tv)
    print(descision)
    return render_template('tv_uhf.html',**context)

 
@app.route('/alarmas', methods=['GET', 'POST'])
@login_required
def alarmas():
    user_ip = session.get('user_ip')
    username = current_user.id
    alarmas = get_alarma_fm(user_id=username)
    
    context = {
        'username': username,
        'alarmas': alarmas,

    }
    print(alarmas)
    return render_template('alarmas.html',**context)


@app.route('/pruebas', methods=['GET', 'POST'])
@login_required
def pruebas():
    user_ip = session.get('user_ip')
    username = current_user.id
    alarmas = get_alarma_fm(user_id=username)
    
    context = {
        'username': username,
        'alarmas': alarmas,

    }
    return render_template('pruebas.html',**context)