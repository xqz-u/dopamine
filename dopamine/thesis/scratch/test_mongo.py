import datetime
import pprint

from pymongo import MongoClient


def main():
    client = MongoClient("mongodb://localhost:27017/")

    db = client.test_database
    # databases and collections are created lazily aka when a document is
    # first inserted, so "test_database" should not exist yet
    client.list_database_names()

    collection = db.test_collection
    # same as above comment goes here
    db.list_collection_names()

    # add some documents with variable schemas
    new_posts = [
        {
            "author": "Mike",
            "text": "Another post!",
            "tags": ["bulk", "insert"],
            "date": datetime.datetime(2009, 11, 12, 11, 14),
        },
        {
            "author": "Eliot",
            "title": "MongoDB is fun",
            "text": "and pretty easy too!",
            "date": datetime.datetime(2009, 11, 10, 10, 45),
        },
    ]

    results = collection.insert_many(new_posts)
    results.inserted_ids

    # now the database "test_database" and the collection
    # "test_collection" exist
    assert "test_database" in client.list_database_names()
    assert "test_collection" in db.list_collection_names()

    # example query with some filtering
    for post in collection.find({"author": "Mike"}):
        pprint.pprint(post)


# main()
