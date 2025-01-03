def get_start_banner():
    banner = [
        '\n' + '='*70,
        '                 风景园林平面图描图机 v1.0',
        '                          (≧▽≦)',
        '                 @地球研究所ppt @罗晓敏 @陈博士开发',
        '\n' + '='*70,
        '  本工具专注于【设计数据合成】',
        '  本团队专注于【人工智能辅助设计】、【设计智能化解决方案】',
        '='*70,
        '  联系我们：',
        '  - 微信咨询：15690576620（罗晓敏）/7053677787（陈博士）',
        '  - 小红书/B站：地球研究所PPT',
        '  - 微信公众号：地球研究社PPT',
        '='*70 + '\n'
    ]
    return '\n'.join(banner)

def get_end_banner():
    return get_start_banner()  # 使用相同的标签

# 为了增加一点保护，可以添加一个简单的检查
def _verify_authenticity():
    """简单的验证函数，确保文件未被修改"""
    import hashlib
    # 计算标签内容的哈希值
    content = get_start_banner()
    hash_value = hashlib.md5(content.encode()).hexdigest()
    # 这个哈希值应该和预期的值匹配
    expected_hash = "请替换为实际的哈希值"  # 这里可以放置实际内容的哈希值
    return hash_value 