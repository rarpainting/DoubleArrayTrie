package darts

import (
	"testing"
)

var (
	words = []string{"一举", "一举一动", "一举成名", "一举成名天下知", "万能", "万能胶"}

	Dat *DoubleArrayTrie
)

func TestTrie1(t *testing.T) {
	Dat = NewDoubleArrayTrie()
	if Dat == nil {
		t.Fatal("Dat is nil")
	}
	Dat.Build(words)
}

func BenchmarkTrie1(b *testing.B) {
	for i := b.N; i > 0; i-- {
		dat := NewDoubleArrayTrie()
		dat.Build(words)
		dat.ExactMatchSearch("万能")
		dat.ExactMatchSearch("哈哈")
		dat.CommonPrefixSearch("一举成名天下知")
	}
}

func BenchmarkTrie0(b *testing.B) {
	for i := b.N; i > 0; i-- {
		dat := NewDoubleArrayTrie()
		dat.Build(words)
	}
}

func BenchmarkTrie2(b *testing.B) {
	for i := b.N; i > 0; i-- {
		Dat.ExactMatchSearch("万能")
	}
}

func BenchmarkTrie3(b *testing.B) {
	for i := b.N; i > 0; i-- {
		Dat.ExactMatchSearch("哈哈")
	}
}

func BenchmarkTrie4(b *testing.B) {
	for i := b.N; i > 0; i-- {
		Dat.CommonPrefixSearch("一举成名天下知")
	}
}
