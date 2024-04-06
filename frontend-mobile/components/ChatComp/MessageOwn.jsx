import { Text, View, StyleSheet } from "react-native";
import TimeAgo from "react-native-timeago";

export default function MessageOwn({ item }) {
  return (
    <View>
      <View style={styles.messageView} key={item._id}>
        <Text style={{ fontSize: 16, color: "#fff" }} key={item._id}>
          {item.text}
        </Text>

        <View style={styles.rightArrow}></View>

        <View style={styles.rightArrowOverlap}></View>
      </View>
      <TimeAgo style={styles.timeText} time={item.createdAt} />
    </View>
  );
}

const styles = StyleSheet.create({
  messageView: {
    backgroundColor: "#FF8216",
    padding: 10,
    marginRight: "5%",
    maxWidth: "60%",
    alignSelf: "flex-end",
    borderRadius: 20,
  },
  timeText: {
    color: "black",
    alignSelf: "flex-end",
    marginRight: "5%",
    fontSize: 10,
    fontWeight: "semibold",
    marginBottom: 5,
  },
  rightArrow: {
    position: "absolute",
    backgroundColor: "#FF8216",
    width: 20,
    height: 25,
    bottom: 0,
    borderBottomLeftRadius: 25,
    right: -10,
  },

  rightArrowOverlap: {
    position: "absolute",
    backgroundColor: "white",
    width: 20,
    height: 35,
    bottom: -6,
    borderBottomLeftRadius: 18,
    right: -20,
  },
});
