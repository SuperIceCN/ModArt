/*
 * 用于抓取 https://mcmod.cn 上的贴图链接，仅供自用，侵删。
 */
const data = {}
for (const each of $0.children) {
    const ele = each.children[0].children[0];
    const name = ele.getAttribute("alt");
    const index = name.indexOf(" ");
    data[name.substring(index + 2, name.length - 1)] = {
        chn: name.substring(0, index),
        url: "https:" + ele.getAttribute("src")
    }
}
console.log(JSON.stringify(data));