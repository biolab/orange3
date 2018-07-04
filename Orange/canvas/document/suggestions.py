class Suggestions:
    def __init__(self, scheme):
        self.__scheme = scheme
        self.links_added = []
        self.__scheme.onNewLink(self.new_link)

    def new_link(self, link):
        self.links_added.append(link)
        print(self.links_added)
