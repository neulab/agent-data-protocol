import pytest
from raw_to_standardized import parse_action


def test_basic_actions():
    # Test simple string arguments
    func_name, kwargs = parse_action("click('123')")
    assert func_name == "click"
    assert kwargs == {"bid": "123"}

    # Test multiple string arguments
    func_name, kwargs = parse_action('fill("456", "hello world")')
    assert func_name == "fill"
    assert kwargs == {"bid": "456", "value": "hello world"}


def test_numeric_arguments():
    # Test integer arguments
    func_name, kwargs = parse_action("scroll(100, -200)")
    assert func_name == "scroll"
    assert kwargs == {"delta_x": 100, "delta_y": -200}

    # Test float arguments
    func_name, kwargs = parse_action("noop(1000.5)")
    assert func_name == "noop"
    assert kwargs == {"wait_ms": 1000.5}

    # Test negative float
    func_name, kwargs = parse_action("scroll(-100.5, 200.75)")
    assert func_name == "scroll"
    assert kwargs == {"delta_x": -100.5, "delta_y": 200.75}


def test_list_arguments():
    # Test list of strings
    func_name, kwargs = parse_action('select_option("123", ["option1", "option2"])')
    assert func_name == "select_option"
    assert kwargs == {"bid": "123", "options": ["option1", "option2"]}

    # Test list with mixed types
    func_name, kwargs = parse_action('upload_file("123", ["file1.txt", "file2.txt"])')
    assert func_name == "upload_file"
    assert kwargs == {"bid": "123", "file": ["file1.txt", "file2.txt"]}


def test_keyword_arguments():
    # Test with keyword arguments
    func_name, kwargs = parse_action('click("123", button="right", modifiers=["Control", "Shift"])')
    assert func_name == "click"
    assert kwargs == {"bid": "123", "button": "right", "modifiers": ["Control", "Shift"]}

    # Test with default keyword arguments
    func_name, kwargs = parse_action('click("123", button="middle")')
    assert func_name == "click"
    assert kwargs == {"bid": "123", "button": "middle"}


def test_special_values():
    # Test None
    func_name, kwargs = parse_action('select_option("123", None)')
    assert func_name == "select_option"
    assert kwargs == {"bid": "123", "options": None}

    # Test boolean values
    func_name, kwargs = parse_action("click(True)")
    assert func_name == "click"
    assert kwargs == {"bid": True}


if __name__ == "__main__":
    pytest.main([__file__])
