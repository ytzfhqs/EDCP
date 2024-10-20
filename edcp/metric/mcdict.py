class McDict(dict):
    def __setitem__(self, key, value):
        # 调用父类的__setitem__方法，插入键值对
        super().__setitem__(key, value)
        # 按照要求对字典的键进行排序
        self._sort_dict()

    def _sort_dict(self):
        # 如果字典中有'text'键，则将其放到首位
        items = list(self.items())
        items.sort(key=lambda x: (x[0] != 'text', x[0]))
        # 清空当前字典并重新插入排序后的键值对
        super().clear()
        super().update(items)

    def update(self, *args, **kwargs):
        # 继承并覆盖update方法，确保更新后按顺序排列
        super().update(*args, **kwargs)
        self._sort_dict()
