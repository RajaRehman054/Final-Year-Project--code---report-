import { Text, View, StyleSheet } from "react-native";
import TimeAgo from "react-native-timeago";

export default function MessageOther({ item }) {
  return (
    <View>
      <View style={styles.messageView} key={item._id}>
        <Text
          style={{ fontSize: 16, color: "#000", justifyContent: "center" }}
          key={item._id}
        >
          {item.text}
        </Text>
        <View style={styles.leftArrow}></View>
        <View style={styles.leftArrowOverlap}></View>
      </View>
      <TimeAgo style={styles.timeText} time={item.createdAt} />
    </View>
  );
}

const styles = StyleSheet.create({
  messageView: {
    backgroundColor: "#dedede",
    padding: 10,
    marginLeft: "5%",
    maxWidth: "60%",
    alignSelf: "flex-start",
    borderRadius: 20,
  },
  timeText: {
    color: "black",
    alignSelf: "flex-start",
    marginLeft: "5%",
    fontSize: 10,
    fontWeight: "semibold",
    marginBottom: 5,
  },
  leftArrow: {
    position: "absolute",
    backgroundColor: "#dedede",
    width: 20,
    height: 25,
    bottom: 0,
    borderBottomRightRadius: 25,
    left: -10,
  },

  leftArrowOverlap: {
    position: "absolute",
    backgroundColor: "white",
    width: 20,
    height: 35,
    bottom: -6,
    borderBottomRightRadius: 18,
    left: -20,
  },
});
