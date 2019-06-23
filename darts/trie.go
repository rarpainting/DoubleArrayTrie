/*
状态 base 的转移方程
base[t] + c.code = base[tc]
base[t] + x.code = base[tx]

疑惑:
1. dat 是以 ac 自动机为蓝本的, 为什么不使用 ac-auto-mation 的以 mark 标志为结束标志 ?
2. fetch 两次检查 dat.length
*/
package darts

import (
	"fmt"
	"sort"
)

type Node struct {
	code  int
	depth int
	left  int
	right int
}

func (n *Node) String() string {
	return fmt.Sprintf("node code %d depth %d left %d right %d", n.code, n.depth, n.left, n.right)
}

type ListNode struct {
	size_ int
	nodes []*Node
}

func NewListNode() *ListNode {
	return &ListNode{size_: 0}
}

func (l *ListNode) size() int {
	return l.size_
}

// TODO: check index > size
func (l *ListNode) get(index int) *Node {
	if index < 0 {
		return nil
	}
	return l.nodes[index]
}

func (l *ListNode) add(node *Node) {
	l.nodes = append(l.nodes, node)
	l.size_++
}

type Word struct {
	runes []rune
}

func NewWord(word string) *Word {
	return &Word{runes: []rune(word)}
}

func (w *Word) GetWord() string {
	return string(w.runes)
}

func (w *Word) GetRune(index int) rune {
	return w.runes[index]
}

func (w *Word) GetRunes() []rune {
	return w.runes
}

func (w *Word) Size() int {
	return len(w.runes)
}

func (w *Word) String() string {
	return string(w.runes)
}

type WordCodeDict struct {
	nextCode int
	dict     map[rune]int
}

// 从关键词构建字典
func NewWordCodeDict(words []*Word) *WordCodeDict {
	var _words []rune
	dict := make(map[rune]int)
	for _, word := range words {
		for _, r := range word.GetRunes() {
			dict[r] = 0
		}
	}

	for r := range dict {
		_words = append(_words, r)
	}

	// sort _words
	sort.Sort(ByRune(_words))

	nextCode := 1
	for _, r := range _words {
		dict[r] = nextCode
		nextCode++
	}

	return &WordCodeDict{nextCode: nextCode, dict: dict}
}

func (d *WordCodeDict) Code(word rune) int {
	if code, ok := d.dict[word]; ok {
		return code
	}
	// 返回一个非法值(新值), 因为 d.nextCode > len(d.dict)
	return d.nextCode
}

type DoubleArrayTrie struct {
	base         []int
	check        []int
	used         []bool
	size         int
	allocSize    int
	key          []*Word // 词列表
	keySize      int
	length       []int
	value        []int
	progress     int
	nextCheckPos int
	error_       int
	// 构建字典树, 记录 字 在 整个字典 的顺序号(从 1 开始, nextCode 为最大数+1)
	wordCodeDict *WordCodeDict
}

func (dat *DoubleArrayTrie) resize(newSize int) int {
	base2 := make([]int, newSize)
	check2 := make([]int, newSize)
	used2 := make([]bool, newSize)

	if dat.allocSize > 0 {
		copy(base2, dat.base)
		copy(check2, dat.check)
		copy(used2, dat.used)
	}
	dat.base = base2
	dat.check = check2
	dat.used = used2
	dat.allocSize = newSize
	return newSize
}

// 建立 trie tree
// 以某个节点为父节点, (单纯)构建该节点的子节点
func (dat *DoubleArrayTrie) fetch(parent *Node, siblings *ListNode) int {
	if dat.error_ < 0 {
		return 0
	}
	prev := 0

	// if (dat.length != nil ? dat.length[i]:len(key[i]) < parent.depth)
	for i := parent.left; i < parent.right; i++ {
		// 非法单词过滤
		if dat.length != nil && dat.length[i] != 0 {
			continue
		} else if dat.key[i].Size() < parent.depth {
			// 子节点的长度必须大于父节点的深度(即单词长度)
			// 如果 len(dat.key[i]) < parent.depth, 说明已经是叶子节点了
			continue
		}

		tmp := dat.key[i]
		cur := 0
		if dat.length != nil && dat.length[i] != 0 {
			// 查询 dat.key[i] 在字典中的序号, 如果不在字典中就给新的序号
			cur = dat.wordCodeDict.Code(tmp.GetRune(parent.depth)) + 1
		} else if tmp.Size() != parent.depth {
			cur = dat.wordCodeDict.Code(tmp.GetRune(parent.depth)) + 1
		}

		// key 必须是字典序
		if prev > cur {
			dat.error_ = -3
			return 0
		}

		// 相同前缀的节点对于父节点视为一个子节点
		//   "ab", "acz", "b"
		//              ROOT(d=0,l=0,r=3)
		//           /                     \
		//          [a(d=1,l=0,r=2)]       [b(d=1,l=2,r=3)]
		//          /               \                   /
		//         [b(d=2,l=0,r=1)] [c(d=2,l=1,r=2)]  [nil(d=2,l=2,r=3)]
		//         /                    /
		//        [nil(d=3,l=0,r=1)]   [z(d=3,l=1,r=2)]
		//                             /
		//                           [nil(d=4,l=1,r=2)]
		// 一个完整的单词最后一个结束节点的 left, right 与父节点保持一致
		if cur != prev || siblings.size() == 0 {
			tmpNode := &Node{
				depth: parent.depth + 1,
				code:  cur,
				left:  i, // 左边界根据不同的前缀而不同
			}

			if siblings.size() != 0 {
				// 新的节点要加入, 前一个右节点的边界需要调整, 与新节点的左边界相同
				siblings.get(siblings.size() - 1).right = i
			}

			siblings.add(tmpNode)
		}

		prev = cur
	}
	// 父节点的子节点构建完成
	if siblings.size() != 0 {
		// 右边界与父节点相同
		siblings.get(siblings.size() - 1).right = parent.right
	}
	return siblings.size()
}

func (dat *DoubleArrayTrie) insert(siblings *ListNode) int {
	if dat.error_ < 0 {
		return 0
	}

	begin := 0
	nonzero_num := 0
	first := 0 // 第一轮循环的标识
	var pos int

	if siblings.get(0).code+1 > dat.nextCheckPos { // last position
		pos = siblings.get(0).code + 1
	} else {
		pos = dat.nextCheckPos
	}
	pos -= 1

	if dat.allocSize <= pos {
		dat.resize(pos + 1)
	}
OUTER:
	// 此循环体的目标是找出满足 base[begin + (a1...an)]==0, check[begin + (a1...an)]==0 的 n 个空闲空间, (a1...an) 是 siblings 中的 n 个节点
	for {
		pos++

		if dat.allocSize <= pos {
			dat.resize(pos + 1)
		}
		if dat.check[pos] != 0 {
			nonzero_num++
			continue
		} else if first == 0 { // 第一轮循环
			dat.nextCheckPos = pos
			first = 1
		}

		// 当前位置离第一个兄弟节点的距离
		begin = pos - siblings.get(0).code

		if dat.allocSize <= (begin + siblings.get(siblings.size()-1).code) {
			// progress can be zero
			var l float64
			tmp_l := 1.0 * float64(dat.keySize) / float64(dat.progress+1)
			if 1.05 > tmp_l {
				l = 1.05
			} else {
				l = tmp_l
			}
			dat.resize(int(float64(dat.allocSize) * l))
		}

		// 这个位置已经被使用了
		if dat.used[begin] {
			continue
		}

		// 检查是否存在冲突
		// 如果 check[i] 不为 0, 则说明此位置已经被别的状态占领了, 需要更换到下一个位置
		for i := 0; i < siblings.size(); i++ {
			if dat.base[begin+siblings.get(i).code] != 0 {
				continue OUTER
			}
			if dat.check[begin+siblings.get(i).code] != 0 {
				continue OUTER
			}
		}
		// 找到一个没有冲突的位置
		break
	}

	// pos-dat.nextCheckPos >= 0 恒成立
	if 1.0*float64(nonzero_num)/float64(pos-dat.nextCheckPos+1) >= 0.95 {
		dat.nextCheckPos = pos
	}
	// 标记位置被占用
	dat.used[begin] = true
	tmp_size := begin + siblings.get(siblings.size()-1).code + 1
	// 更新 tire 的 size
	if dat.size < tmp_size {
		dat.size = tmp_size
	}

	// base[s] + c = t
	// check[t] = s
	for i := 0; i < siblings.size(); i++ {
		dat.check[begin+siblings.get(i).code] = begin
	}

	// 计算所有子节点的 base
	for i := 0; i < siblings.size(); i++ {
		new_siblings := NewListNode()
		//// 一个词的终止且不为其他词的前缀, 其实就是叶子节点
		if dat.fetch(siblings.get(i), new_siblings) == 0 {
			if dat.value != nil {
				dat.base[begin+siblings.get(i).code] = dat.value[siblings.get(i).left-1]*(-1) - 1
			} else {
				dat.base[begin+siblings.get(i).code] = siblings.get(i).left*(-1) - 1
			}

			if dat.value != nil && (dat.value[siblings.get(i).left]*(-1)-1) >= 0 {
				dat.error_ = -2
				return 0
			}

			dat.progress++
		} else {
			h := dat.insert(new_siblings)
			dat.base[begin+siblings.get(i).code] = h
		}
	}

	return begin
}

func NewDoubleArrayTrie() *DoubleArrayTrie {
	return &DoubleArrayTrie{}
}

func (dat *DoubleArrayTrie) clear() {
	*dat = DoubleArrayTrie{}
}

// 减肥 base / check
// dat.allocSize = dat.size
func (dat *DoubleArrayTrie) loseWeight() {
	base2 := make([]int, dat.size)
	check2 := make([]int, dat.size)

	if dat.allocSize > 0 {
		copy(base2, dat.base)
		copy(check2, dat.check)
	}
	dat.base = base2
	dat.check = check2
	dat.allocSize = dat.size
	return
}

func (dat *DoubleArrayTrie) GetSize() int {
	return dat.size
}

func (dat *DoubleArrayTrie) GetNonzeroSize() int {
	result := 0
	for i := 0; i < dat.size; i++ {
		if dat.check[i] != 0 {
			result++
		}
	}
	return result
}

func (dat *DoubleArrayTrie) Build(_key []string) int {
	return dat.BuildAdvanced(_key, nil, nil, len(_key))
}

func (dat *DoubleArrayTrie) BuildAdvanced(_key []string, _length []int, _value []int, _keySize int) int {
	if _keySize > len(_key) || _key == nil {
		return 0
	}
	var words []*Word
	for _, key := range _key {
		words = append(words, NewWord(key))
	}
	dat.key = words
	dat.length = _length
	dat.keySize = _keySize
	dat.value = _value
	dat.progress = 0
	dat.wordCodeDict = NewWordCodeDict(words)

	// 32个双字节
	dat.resize(65536 * 32)

	dat.base[0] = 1
	dat.nextCheckPos = 0

	root_node := &Node{
		left:  0,
		right: dat.keySize,
		depth: 0,
	}

	siblings := NewListNode()
	// Root 节点是 Null 节点
	dat.fetch(root_node, siblings)
	dat.insert(siblings)

	dat.key = nil
	dat.used = nil
	dat.loseWeight()
	return dat.error_
}

func (dat *DoubleArrayTrie) ExactMatchSearch(key string) int {
	return dat.ExactMatchSearchAdvanced(key, 0, 0, 0)
}

func (dat *DoubleArrayTrie) ExactMatchSearchAdvanced(key string, pos int, length int, nodePos int) int {
	word := NewWord(key)
	if length <= 0 {
		length = word.Size()
	}
	if nodePos <= 0 {
		nodePos = 0
	}

	var result = -1

	keyChars := word.GetRunes()
	b := dat.base[nodePos]
	var p int
	for i := pos; i < length; i++ {
		p = b + dat.wordCodeDict.Code(keyChars[i]) + 1
		if b == dat.check[p] {
			b = dat.base[p]
		} else {
			return result
		}
	}

	p = b
	n := dat.base[p]

	if b == dat.check[p] && n < 0 {
		result = n*(-1) - 1
	}
	return result
}

func (dat *DoubleArrayTrie) CommonPrefixSearch(key string) []int {
	return dat.CommonPrefixSearchAdvanced(key, 0, 0, 0)
}

func (dat *DoubleArrayTrie) CommonPrefixSearchAdvanced(key string, pos int, length int, nodePos int) []int {
	word := NewWord(key)
	if length <= 0 {
		length = word.Size()
	}
	if nodePos <= 0 {
		nodePos = 0
	}

	var result []int
	keyChars := word.GetRunes()
	b := dat.base[nodePos]
	var n, p int

	for i := pos; i < length; i++ {
		p = b
		n = dat.base[p]

		if b == dat.check[p] && n < 0 {
			result = append(result, (n*(-1) - 1))
		}

		p = b + dat.wordCodeDict.Code(keyChars[i]) + 1
		if b == dat.check[p] {
			b = dat.base[p]
		} else {
			return result
		}
	}

	p = b
	n = dat.base[p]

	if b == dat.check[p] && n < 0 {
		result = append(result, (n*(-1) - 1))
	}
	return result
}

func (dat *DoubleArrayTrie) Dump() (str string) {
	for i := 0; i < dat.size; i++ {
		str = str + fmt.Sprintf("i: %d", i)
		str = str + fmt.Sprintf(" [%d", dat.base[i])
		str = str + fmt.Sprintf(", %d]\n", dat.check[i])
	}

	return str
}

type ByRune []rune

func (a ByRune) Len() int           { return len(a) }
func (a ByRune) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByRune) Less(i, j int) bool { return a[i] < a[j] }
