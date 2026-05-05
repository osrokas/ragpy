from parsers import *

def main():
    parser = JsonParser(json_data)
    document = parser.get_document()
    elements = parser.filter_elements_by_type(["text", "caption", "figure"])

if __name__ == "__main__":
    main()
