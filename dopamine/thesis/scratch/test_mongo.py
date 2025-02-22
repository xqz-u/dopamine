import datetime
import pprint

from pymongo import MongoClient


def main(mongo_uri: str = "mongodb://localhost:27017/"):
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=int(5e3))

    db = client.test_database
    # databases and collections are created lazily aka when a document is
    # first inserted, so "test_database" should not exist yet
    print(client.list_database_names())

    collection = db.test_collection
    # same as above comment goes here
    print(db.list_collection_names())

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
    print(results.inserted_ids)

    # now the database "test_database" and the collection
    # "test_collection" exist
    assert "test_database" in client.list_database_names()
    assert "test_collection" in db.list_collection_names()

    # example query with some filtering
    for post in collection.find({"author": "Mike"}):
        pprint.pprint(post)


if __name__ == "__main__":
    main()
    # main("mongodb://s3680622:<pswd>@peregrine.hpc.rug.nl:27017/")
    # main("mongodb://s3680622:<pswd>@129.125.60.96:27017/")
    # main("mongodb://129.125.60.96:27017/")
    # main("mongodb://peregrine.hpc.rug.nl:27017/")

