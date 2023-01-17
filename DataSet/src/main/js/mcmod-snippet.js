const data = {}
for (const each of $0.children) {
    const ele = each.children[0].children[0];
    const name = ele.getAttribute("alt");
    const index = name.indexOf(" ");
    data[name.substring(index + 2, name.length - 1)] = {
        chn: name.substring(0, index),
        url: ele.getAttribute("src")
    }
}
console.log(data);