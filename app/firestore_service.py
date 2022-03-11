import firebase_admin 
from firebase_admin import credentials
from firebase_admin import firestore

credentials = credentials.ApplicationDefault()
firebase_admin.initialize_app(credentials)

db = firestore.client() # nueva instancia del cliente de firestore 

def get_users():
    return db.collection('users').get()

def get_user(user_id):
    return db.collection('users').document(user_id).get()



def user_put(user_data):
    user_ref=db.collection('users').document(user_data.username)
    user_ref.set({ 'password': user_data.password })


def get_todos(user_id):
    return db.collection('users').document(user_id).collection('todos').get()

def put_todo(user_id, description):
    todos_collection_ref = db.collection('users').document(user_id).collection('todos').document()
    todos_collection_ref.set({'descripcion': description, 'Done': False})

def delete_todo(user_id, todo_id):
    todo_ref = db.document('users/{}/todos/{}'.format(user_id, todo_id))
    #todo_ref = db.collection('users').document(user_id).collection('todos').document(todo_id)
    todo_ref.delete()

def update_todo(user_id, todo_id, done):
    pass

def get_todo_ref(user_id, todo_id):
    return db.document('users/{}/todos/{}'.format(user_id, todo_id))
    