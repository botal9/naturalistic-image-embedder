from html.parser import HTMLParser


class HdrHtmlParser(HTMLParser):
    def __init__(self):
        self.links = []
        self.additional_info = []
        self._inside_parameters_table = False
        super().__init__()

    def reset(self):
        self.links = []
        self.additional_info = []
        self._inside_parameters_table = False
        super().reset()

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[0] == 'href':
                    if not self._validate_link(attr[0]):
                        self.links.append(attr[1])
        elif tag == 'table':
            for attr in attrs:
                if attr[0] == 'id' and attr[1] == 'parameterstable':
                    self._inside_parameters_table = True

    def handle_endtag(self, tag):
        if tag == 'table' and self._inside_parameters_table:
            self._inside_parameters_table = False

    def handle_data(self, data):
        if self._inside_parameters_table:
            stripped_data = str(data).strip()
            if len(stripped_data) > 0:
                self.additional_info.append(stripped_data)

    def parse(self, html):
        self.feed(html)

    def _validate_link(self, link):
        return link in self.links or '#' in link or 'javascript:' in link
