import couchdb

def couchDBconnection():
    """
    This function allows to create a CouchDB connection.
    """

    couch = couchdb.Server('') #DB http
    database = couch['irb']
    print('Connected to CouchDB')

    return database

