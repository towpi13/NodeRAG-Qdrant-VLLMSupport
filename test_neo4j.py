from neo4j import GraphDatabase

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j+s://e4731256.databases.neo4j.io"
AUTH = ("neo4j", "EgLVdLWxDRqTGQzsaGiXk1oUP6h8OA_kbexW8aY3aiY")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        # session.run("CREATE (n:Test1 {name: 'Firssdfsdft Node'})")
        result = session.run("MATCH (n:Test1) RETURN n.name AS name")
        print(result.single()["name"])


